from argparse import ArgumentParser
from pathlib import Path
import os,sys
import torch
import logging
import json
import random
import numpy as np
import pandas as pd
from collections import namedtuple
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import pdb
from hybridmf import *
from inference import *
import torch.nn.functional as F
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertForMaskedLM
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from util import MAX_TURN, PREVENT_FACTOR, PROMOTE_FACTOR, PREVENT_LIST, REDUCE_LIST, STOP_LIST, boolean_string
import copy
import evaluate
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')
NUM_PAD = 3

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids ")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)

sys.path.append("/pytorch_code")

def inference_now(target_items,model,tokenizer,keyfile,description,num_item,device,mode):
    sep_tok = tokenizer.vocab['[SEP]']
    cls_tok = tokenizer.vocab['[CLS]']
    pad_tok = tokenizer.vocab['[PAD]']

    epoch_dataset = PregeneratedDataset(epoch=0, training_path=keyfile, tokenizer=tokenizer, max_seq_len = 256, sep=" ", no_ins_at_first = False, num_data_epochs=1)
    epoch_sampler = SequentialSampler(epoch_dataset)
    generate_dataloader = DataLoader(epoch_dataset, sampler=epoch_sampler,batch_size=1)
    prevent = [ tokenizer.vocab.get(x) for x in PREVENT_LIST]
    reduce_l = REDUCE_LIST |  STOP_LIST
    reduces = [ tokenizer.vocab.get(x) for x in reduce_l]  
    reduces = [s for s in reduces if s]
    
    new_text = []
    with tqdm(total=len(generate_dataloader), desc=f"Epoch {0}") as pbar:
        for step, batch in enumerate(generate_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, no_ins = batch
            if mode==0:
                predict_ids = greedy_search(model, input_ids, segment_ids, input_mask, no_ins = no_ins,tokenizer=tokenizer, prevent=prevent, reduce= reduces)
            else:
                predict_ids = sample_generate(model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=0.8, tokenizer=tokenizer, prevent=prevent, reduce= reduces)
 
            output =  " ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in predict_ids[0].detach().cpu().numpy() if x!=sep_tok and x != pad_tok and x != cls_tok])
            output = output.replace(" ##", "") 
            cur_str = output.split(' ')
            if len(cur_str)>100:
                output = ' '.join(cur_str[:100])
            new_text.append(output)
    
    text_all = [] 
    count=0
    for item in range(num_item):
        if item in target_items:
#            text_all.append(description[item] + ' ' +  new_text[count])
            text_all.append(new_text[count])
            count+=1
        else:
            text_all.append(description[item])
    return text_all,new_text

def evaluation(MF_model,target_items,negatives, orig_emb,cur_text,test,device,rating_matrix):
    text_emb_all = sbert.encode(cur_text,convert_to_tensor=True,batch_size=4096).to(device)
    avg_rating,avg_HITS,avg_rank = 0,0,0
    with torch.no_grad():
        item_emb = rating_matrix[:,negatives].t()
        text_emb = text_emb_all[negatives]
        item_to_idx = {item:idx for (idx,item) in enumerate(negatives)}
        locations = [item_to_idx[item] for item in target_items]
        for user in range(rating_matrix.shape[0]):
            user_emb = rating_matrix[user,:].expand(len(negatives),-1)
            predictions = MF_model(user_emb,item_emb,text_emb).detach().cpu().numpy()
            ranks = np.argsort(-predictions)
            intersect,x_ind,y_ind = np.intersect1d(ranks,locations,return_indices=True)
            avg_rank += sum(x_ind+1) 
            avg_rating += sum(predictions[locations])
            avg_HITS += len(np.intersect1d(ranks[:20],locations))
        avg_rating/=len(target_items)*rating_matrix.shape[0]
        avg_HITS/=rating_matrix.shape[0]*20
        avg_rank/=len(target_items)*rating_matrix.shape[0]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        cur_words = []
        for item in target_items:
            cur_words.append(len(cur_text[item].split(' ')))
        avg_text_sim = torch.mean(cos(orig_emb[target_items],text_emb_all[target_items])).item()
 
    return avg_rating,avg_HITS,avg_rank,avg_text_sim,np.average(cur_words)

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

def convert_example(example, tokenizer, max_seq_length, args=None):
    tokens = example["tokens"]
    lm_label_tokens = example["lm_label_tokens"]

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
        lm_label_tokens = lm_label_tokens[:max_seq_length]

    assert len(tokens) == len(lm_label_tokens) <= max_seq_length  # The preprocessed data should be already truncated
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    lm_label_ids = tokenizer.convert_tokens_to_ids(lm_label_tokens)

    input_array = np.zeros(max_seq_length, dtype=int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=bool)

    lm_label_array = np.full(max_seq_length, dtype=int, fill_value=-1)
    lm_label_array[:min(len(lm_label_ids) + NUM_PAD,max_seq_length) ] = 0
    lm_label_array[:len(lm_label_ids)] = lm_label_ids

    if args.wp:
        cls_pos = tokens.index('[CLS]')
        lm_label_array[:cls_pos] = -1

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             )
    return features,example["id"]


class Dataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, args=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            
        self.item_ids = np.zeros(shape=(num_samples),dtype=int)
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                example = json.loads(line)
                features,item_id = convert_example(example, tokenizer, seq_len, args=args)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                self.item_ids[i] = item_id
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                self.item_ids[item],
                )


def obtain_text_emb(model,tokenizer,device,raw_text,batch_size,emb_dim):
    batch_num = (len(raw_text)-1)//batch_size+1
    text_emb_all = torch.zeros((len(raw_text),emb_dim)).to(device)
    for i in range(batch_num):
        torch.cuda.empty_cache() 
        st_idx,ed_idx = i*batch_size,(i+1)*batch_size
        if ed_idx>len(raw_text):
            ed_idx = len(raw_text)
        text_ids,masks = [],[]
        for j in range(st_idx,ed_idx):
            cur_ids,mask = np.zeros(256),np.zeros(256)
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text[j]))[:256]
            cur_ids[:len(ids)] = ids
            mask[:len(ids)]=1
            text_ids.append(cur_ids)
            masks.append(mask)
        text_ids,masks = torch.LongTensor(np.array(text_ids)).to(device),torch.LongTensor(np.array(masks)).to(device)
        text_emb = torch.mean(model(text_ids,attention_mask = masks)[1][-1],dim=1)
        text_emb_all[st_idx:ed_idx] = text_emb
        del text_ids,masks

    return text_emb_all


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--coeff', type=float, default = 0.01)
    parser.add_argument('--output', type=str)
    parser.add_argument('--mode', type=str) 
    parser.add_argument('--mf_model_path', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_steps", 
                        default=0, 
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument("--wp", type=bool, default=False, help="if train on wp")
    parser.add_argument('--from_scratch', action='store_true', help='do not load prtrain model, only random initialize')
    parser.add_argument("--output_step", type=int, default=100000, help="Number of step to save model")
    parser.add_argument("--target_ratio", type=float, default=0.01, help="Number of step to save model")
    parser.add_argument('--noi_decay',
                        type=int,
                        default=1,
                        help="round number to decay NOI prob") 
    parser.add_argument('--reduce_decay',
                        type=int,
                        default=1,
                        help="round number to decay reduce prob") 
    parser.add_argument('--verbose', type=int,
                        default=0,
                        help="verbose level") 
    parser.add_argument('--prevent', 
                        type=bool, 
                        default=True,
                        help="avoid generating several words")
    parser.add_argument('--reduce_stop',
                        type=bool, 
                        default=True, 
                        help="reduce stopwords")    
    parser.add_argument('--lessrepeat',
                        type=bool, 
                        default=True, 
                        help="reduce repetition (only for tokenwise)")
    parser.add_argument('--sep',
                         type=str, default=" ", help="token to seperate keywords")
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=256,
                        help="max sequence length") 
    parser.add_argument("--no_ins_at_first", 
                        type=bool, 
                        default=False, 
                        help="Do not insert at the begining of the text")
    args = parser.parse_args()

    f = open(args.output,'w')
 
    set_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = pd.read_csv('data/amazon_rating.tsv',header=None,sep='\t').values   
    users,items = np.unique(X[:,0]),np.unique(X[:,1])
    user_map,item_map = {user:idx for (idx,user) in enumerate(users)},{item:idx for (idx,item) in enumerate(items)}
    rating_matrix = torch.zeros((len(users),len(items)))
    i1,i2,i3 = [],[],[]
    for i in range(X.shape[0]):
        X[i,0] = user_map[X[i,0]]
        X[i,1] = item_map[X[i,1]]
        if np.random.random()<=0.9:
            if np.random.random()<=0.9:
                i1.append(i)
            else:
                i2.append(i)
            rating_matrix[int(X[i,0]),int(X[i,1])]=X[i,2]
        else:
            i3.append(i)
    train,val,test = X[i1],X[i2],X[i3]
    ratings = X
    description = pd.read_csv('data/amazon_description.txt',header=None).values.flatten()
    keywords = pd.read_csv('data/amazon_description.key.txt',header=None).values.flatten() 
    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    samples_per_epoch = []
    num_data_epochs = args.epochs
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break

    bleu = evaluate.load("bleu")
    meteor = evaluate.load('meteor')
    bertscore = evaluate.load("bertscore")
    rating_matrix = rating_matrix.to(device)
    orig,cur,pointer_res = [[],[],[],[],[]],[[],[],[],[],[],[],[],[]],[[],[],[],[],[],[],[]]
    for seed in range(5):
        set_seed(seed)
        MF_model = HMF(ratings,384).to(device)
        MF_model.load_state_dict(torch.load(args.mf_model_path))
        MF_model.eval()
        users,items = np.unique(ratings[:,0]),np.unique(ratings[:,1])
        user_map,item_map = {user:idx for (idx,user) in enumerate(users)},{item:idx for (idx,item) in enumerate(items)}
        num_user,num_item = len(users),len(items)
        print(num_user,num_item,ratings.shape)
        num_target = int(num_item*args.target_ratio)
     
        preds = sorted(np.random.choice(np.arange(num_item),size=int(0.01*num_item),replace=False)
        target_items = set(preds)
        negatives = sorted(np.append(np.random.choice([item for item in range(num_item) if item not in target_items],size=int(0.1*num_item),replace=False),preds))
      
        bleu_ref = []
        for item in preds:
            bleu_ref.append([description[item]])

        text_emb_all = sbert.encode(description,convert_to_tensor=True,batch_size=4096).to(device)

        with torch.no_grad():
            maxv = torch.zeros(num_user).to(device)
            for i in range(num_user):
                maxv[i] = torch.max(MF_model(rating_matrix[i,:].expand(len(negatives),-1),rating_matrix[:,negatives].t(),text_emb_all[negatives]))
 
        print(torch.mean(maxv),torch.max(maxv),torch.std(maxv))
        model = BertForMaskedLM.from_pretrained(args.bert_model).to(device)
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        total_train_examples = 0
        for i in range(args.epochs):
            # The modulo takes into account the fact that we may loop over limited epochs of data
            total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
       
        num_train_optimization_steps = int(
            total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)
 
        avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len = evaluation(MF_model, preds,negatives,text_emb_all, description,test,device,rating_matrix)
        print('Original performance: (target.rating, target.HITS@20, target.rank, text_similiarity, avg_text_len)= {},{},{},{},{}'.format(avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len),file=f,flush=True)
        orig[0].append(avg_rating)
        orig[1].append(avg_HITS)
        orig[2].append(avg_rank)
        orig[3].append(avg_text_sim)
        orig[4].append(avg_len)

        args.output_mode = "classification"
        
        global_step = 0
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {total_train_examples}")
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)
        model.train()
        
        ones_array = torch.ones((args.train_batch_size,int(0.1*num_user))).to(device)
        target_array = torch.ones((args.train_batch_size,int(0.1*num_user))).to(device)
        loss2 = nn.MarginRankingLoss(margin=0.01)
        pointer,best_text,best_rating,best_model  = [],[],0,0
        cur_path = './'
        keyword_path = cur_path + 'keywords.txt'
        np.savetxt(keyword_path,keywords[preds],fmt='%s')

        for epoch in range(args.epochs+1):
            text_all,new_text = inference_now(target_items,model,tokenizer,Path(keyword_path),description,num_item,device,1)
            results = bleu.compute(predictions=new_text, references=bleu_ref)
            bleu_score = results['bleu']
            results = meteor.compute(predictions=new_text, references=bleu_ref)
            meteor_score = results['meteor']
            results = bertscore.compute(predictions=new_text, references=bleu_ref, lang='en')
            bert_score = np.average(results['f1'])
            avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len = evaluation(MF_model, preds,negatives,text_emb_all, text_all,test,device,rating_matrix)
            print('Epoch = {}\t(target.rating, target.HITS, target.rank, text_similiarity, (bleu,meteor,bertscore), avg_text_len)= {},{},{},{},({},{},{}),{}'.format(epoch,avg_rating,avg_HITS,avg_rank,avg_text_sim,bleu_score,meteor_score,bert_score,avg_len),file=f,flush=True) 
        
            epoch_dataset = Dataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory, args=args)
            train_sampler = RandomSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            pointer_loss,promotion_loss=0,0
            
            avgs = []
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad() 
                input_ids, input_mask, segment_ids, lm_label_ids,original_ids = batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device),batch[4]
                outputs = model(input_ids, segment_ids, input_mask, lm_label_ids,)
                loss = outputs[0]
                pointer_loss += loss.item()

                cur_items = np.random.choice(preds,size=args.train_batch_size,replace=False)
                text_emb_cur = obtain_text_emb(model,tokenizer,device,description[cur_items],args.train_batch_size,384)
                chosen_user = np.random.choice(np.arange(num_user),size=int(0.1*num_user),replace=False)
                target_pred = torch.zeros((args.train_batch_size,int(0.1*num_user))).to(device)
                for idx in range(args.train_batch_size):
                    target_pred[idx] = MF_model(rating_matrix[chosen_user,:],rating_matrix[:,cur_items[idx]].t().expand(len(chosen_user),-1),text_emb_cur[idx].expand(len(chosen_user),-1))
                    target_array[idx] = maxv[chosen_user]
                avgs.append(torch.mean(target_pred).item())
                target_loss = loss2(target_pred,target_array,ones_array) 
                promotion_loss += args.coeff*target_loss.item()
                loss += args.coeff*target_loss
                loss.backward()
                optimizer.step()

            print(pointer_loss,promotion_loss,np.average(avgs),file=f,flush=True)
            logger.info("PROGRESS: {}%".format(round(100 * (epoch + 1) / args.epochs, 4)))
            logger.info("EVALERR: {}%".format(tr_loss))
       
        
        np.savetxt('new_text_seed'+str(seed)+'_HMF.txt',np.array(text_all)[preds],fmt='%s')
        np.savetxt('original_seed'+str(seed)+'_HMF.txt',description[preds],fmt='%s')

        cur[0].append(avg_rating)
        cur[1].append(avg_HITS)
        cur[2].append(avg_rank)
        cur[3].append(avg_text_sim)
        cur[4].append(bleu_score)
        cur[5].append(meteor_score)
        cur[6].append(bert_score)
        cur[7].append(avg_len)

    print('original description result',file=f)
    print(orig,file=f)
    print('Rewrite4Rec result',file=f)
    print(cur,file=f)

if __name__ == '__main__':
    main()
