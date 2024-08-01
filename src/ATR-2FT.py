#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2024-07-31
#Adversarial Text Rewriting for Text-aware Recommender Systems
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset,TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer,LineByLineTextDataset
import evaluate
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset
from itertools import chain
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from unisrec import UniSRec
from data.dataset import UniSRecDataset
from collections import defaultdict
from recbole.utils import FeatureSource, FeatureType
import torch.nn.utils.rnn as rnn_utils
from recbole.data.interaction import Interaction
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import copy
import os
import torch.optim as optim

def full_sort_predict(MF_model, interaction,text_embedding,t_feat):
    item_seq = interaction[MF_model.ITEM_SEQ]
    item_seq_len = interaction[MF_model.ITEM_SEQ_LEN]
    item_emb_list = MF_model.moe_adaptor(text_embedding(item_seq))
    seq_output = MF_model.forward(item_seq, item_emb_list, item_seq_len)
    test_items_emb = MF_model.moe_adaptor(t_feat)

    seq_output = F.normalize(seq_output, dim=-1)
    test_items_emb = F.normalize(test_items_emb, dim=-1)

    scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
    return scores

def evaluation(MF_model,target_items,negatives, orig_emb,cur_text,device,num_user,num_item,sbert,test_dict,cur_items):
    text_emb_all = sbert.encode(cur_text,convert_to_tensor=True,batch_size=1024,normalize_embeddings=True).to(device)
    avg_rank,avg_rating,avg_HITS = 0,0,0
    with torch.no_grad():
        text_embedding = torch.zeros((num_item+1,384)).to(device)
        text_embedding[1:] = text_emb_all
        text_embedding = nn.Embedding.from_pretrained(text_embedding,freeze=False,padding_idx=0)
        full_prediction = []
        item_id_list,item_length = test_dict['item_id_list'],test_dict['item_length']
        test_items_emb = MF_model.moe_adaptor(text_embedding(cur_items))
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        test_size = item_id_list.shape[0]
        for i in range((test_size-1)//4096+1):
            st_idx,ed_idx = i*4096,(i+1)*4096
            if ed_idx>test_size:
                ed_idx = test_size
            item_seq = item_id_list[st_idx:ed_idx]
            item_seq_len = item_length[st_idx:ed_idx]
            item_emb_list = MF_model.moe_adaptor(text_embedding(item_seq))
            seq_output = MF_model.forward(item_seq, item_emb_list, item_seq_len)
            seq_output = F.normalize(seq_output, dim=-1)
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
            full_prediction.append(scores)
        full_prediction = torch.cat(full_prediction).cpu().numpy()

        full_ranking = np.argsort(-full_prediction)
        item_to_idx = {item:idx for (idx,item) in enumerate(negatives)}
        locations = [item_to_idx[item] for item in target_items]
        for user in range(full_ranking.shape[0]):
            ranks = full_ranking[user]
            predictions = full_prediction[user]
            intersect,x_ind,y_ind = np.intersect1d(ranks,locations,return_indices=True)
            avg_rank += sum(x_ind+1)  
            avg_rating += sum(predictions[locations])
            avg_HITS += len(np.intersect1d(ranks[:20],locations))
        avg_rating/=len(target_items)*full_ranking.shape[0]
        avg_HITS/=full_ranking.shape[0]*20
        avg_rank/=len(target_items)*full_ranking.shape[0]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        cur_words = []
        for item in target_items:
            cur_words.append(len(cur_text[item].split(' ')))
        avg_text_sim = torch.mean(cos(orig_emb[target_items],text_emb_all[target_items])).item()
 
    print(avg_rating,avg_HITS,avg_rank,avg_text_sim,np.average(cur_words))
    return avg_rating,avg_HITS,avg_rank,avg_text_sim,np.average(cur_words)


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

def obtain_text_emb(model,tokenizer,device,raw_text,batch_size,emb_dim,grad):
    batch_num = (len(raw_text)-1)//batch_size+1
    text_emb_all = torch.zeros((len(raw_text),emb_dim)).to(device)
    if grad==0:
        with torch.no_grad():
            for i in range(batch_num):
                st_idx,ed_idx = i*batch_size,(i+1)*batch_size
                if ed_idx>len(raw_text):
                    ed_idx = len(raw_text)
                text_ids = tokenizer(list(raw_text[st_idx:ed_idx]), add_special_tokens=True, padding=True, truncation=True, max_length=256)["input_ids"]
                text_ids = torch.LongTensor(text_ids).to(device)
                forward_output = model(text_ids,return_dict=True,output_hidden_states=True).hidden_states[-1]
                for j in range(forward_output.shape[0]):  
                    text_emb_all[st_idx+j] = torch.mean(forward_output[j,text_ids[j]!=tokenizer.pad_token_id,:],dim=0)
                del text_ids,forward_output
                torch.cuda.empty_cache() 
    else:
         for i in range(batch_num):
            st_idx,ed_idx = i*batch_size,(i+1)*batch_size
            if ed_idx>len(raw_text):
                ed_idx = len(raw_text)
            text_ids = tokenizer(list(raw_text[st_idx:ed_idx]), add_special_tokens=True, truncation=True, padding=True, max_length=256)["input_ids"]
            text_ids = torch.LongTensor(text_ids).to(device)
            forward_output = model(text_ids,return_dict=True,output_hidden_states=True).hidden_states[-1]
            for j in range(forward_output.shape[0]):  
                text_emb_all[st_idx+j] = torch.mean(forward_output[j,text_ids[j]!=tokenizer.pad_token_id,:],dim=0)
            del text_ids,forward_output
            torch.cuda.empty_cache()

    return text_emb_all


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):           
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
        self.relu = nn.ReLU()
                                                    
    def forward(self, x):                                            
        x = self.fc1(x)                 
        x = self.relu(x)               
        x = self.fc2(x)
        x = nn.functional.normalize(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--coeff', type=float, default = 0.01)
    parser.add_argument('--output', type=str)
    parser.add_argument('--mode', type=str) 
    parser.add_argument('--mf_model_path', type=Path, required=True)
    parser.add_argument('--bert_model', type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to fine-tune for")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for fine-tuning.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=256,
                        help="max sequence length") 
    parser.add_argument("--target_ratio", type=float, default=0.01, help="target item ratio")

    args = parser.parse_args()

     
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Create a DataCollator for Language Modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    bleu = evaluate.load("bleu")
    meteor = evaluate.load('meteor')
    sbert = SentenceTransformer('all-MiniLM-L6-v2',device=device)
    X = pd.read_csv('src/dataset/downstream/amazon_book/amazon_rating.tsv',header=None,sep='\t').values   
    users_all,items_all = np.unique(X[:,0]),np.unique(X[:,1])
    user_map,item_map = {user:idx for (idx,user) in enumerate(users_all)},{item:idx for (idx,item) in enumerate(items_all)}
    ratings = X
    description = pd.read_csv('src/dataset/downstream/amazon_book/amazon_description_new.txt',header=None).values.flatten()
    print(description[0])

    f = open(args.output,'w')
    train_dataset = load_dataset("text",data_files="src/dataset/downstream/amazon_book/amazon_description_new.txt")
    column_names = train_dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
       
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True,remove_columns=column_names)
    # Create a DataCollator for Language Modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    def group_texts(examples):
        block_size=512
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
    )

    train_dataset = lm_dataset["train"]
    batch_encoding = tokenizer(list(description), add_special_tokens=True, truncation=True, max_length=256)["input_ids"]
    train_data = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in batch_encoding]
 
    num_data_epochs = args.epochs
    orig,cur,pointer_res = [[],[],[],[],[]],[[],[],[],[],[],[],[]],[[],[],[],[],[]]
    props = ['src/props/UniSRec.yaml', 'src/props/finetune.yaml']

    # configurations initialization
    config = Config(model=UniSRec, dataset='amazon_book', config_file_list=props)
    init_seed(config['seed'], config['reproducibility'])
    # dataset filtering
    dataset = UniSRecDataset(config)
    print(dataset.field2type)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    MF_model = UniSRec(config, train_data.dataset).to(device)

    latest = dataset._load_feat('src/dataset/downstream/amazon_book/amazon_book.latest.inter',  FeatureSource.INTERACTION)
    latest_dict = {}
    for column in latest.columns:
        if column=='user_id' or column=='item_id':
            latest[column] = latest[column].astype(int)
            latest_dict[column] = torch.LongTensor(latest[column].values)
        else:
            seq_data = [torch.LongTensor([int(val) for val in d]) for d in latest[column].values]
            len_data = torch.LongTensor([len(d) for d in latest[column].values])
            latest_dict[column] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            latest_dict['item_length'] = len_data

    latest_interaction = Interaction(latest_dict).to(device)

    test = dataset._load_feat('src/dataset/downstream/amazon_book/amazon_book.test.inter',  FeatureSource.INTERACTION)
    test_dict = {}
    for column in test.columns:
        if column=='user_id' or column=='item_id':
            test[column] = test[column].astype(int)
            test_dict[column] = torch.LongTensor(test[column].values).to(device)
        else:
            seq_data = [torch.LongTensor([int(val) for val in d]) for d in test[column].values]
            len_data = torch.LongTensor([len(d) for d in test[column].values])
            test_dict[column] = rnn_utils.pad_sequence(seq_data, batch_first=True).to(device)
            test_dict['item_length'] = len_data.to(device)
 
    checkpoint = torch.load(args.mf_model_path)
    MF_model.load_state_dict(checkpoint['state_dict'], strict=False)
 
    for seed in range(1):
        set_seed(seed)
        num_user,num_item = len(users_all),len(items_all)
        print(num_user,num_item,ratings.shape)
        model = AutoModelForCausalLM.from_pretrained(args.bert_model).to(device)
        preds = pd.read_csv('result/amazon/target_item_seed'+str(seed)+'.txt',header=None).values.flatten()
        negatives = pd.read_csv('result/amazon/ranking_item_seed'+str(seed)+'.txt',header=None).values.flatten()
        negatives = negatives.astype(int)
        preds = preds.astype(int)
        target_items = set(preds)
        bleu_ref = []

        negatives_gpu = torch.LongTensor(negatives).to(device)+1
        for item in preds:
            bleu_ref.append([description[item]])
     
        text_emb_all = sbert.encode(description,convert_to_tensor=True,batch_size=1024,normalize_embeddings=True)
        avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len = evaluation(MF_model,preds,negatives,text_emb_all, description,device,num_user,num_item,sbert,test_dict,negatives_gpu)
        print('Original performance: (target.rating, target.HITS@20, target.rank, text_similiarity, avg_text_len)= {},{},{},{},{}'.format(avg_rating,avg_HITS,avg_rank, avg_text_sim,avg_len),file=f,flush=True)

        orig[0].append(avg_rating)
        orig[1].append(avg_HITS)
        orig[2].append(avg_rank)
        orig[3].append(avg_text_sim)
        orig[4].append(avg_len)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
       
        loss2 = nn.MarginRankingLoss(margin=0.01)
        pointer,best_text,best_rating,best_model  = [],[],0,0
        indicator = torch.ones((int(0.1*num_user),len(preds))).to(device)
       
        for epoch in range(args.epochs+1):
            # Define the dataloader for training data
            new_text = []
            text_all = copy.deepcopy(description)
            with torch.no_grad():
                model.eval()
                for (idx,item) in enumerate(preds):
                    prompt = description[item]
                    input_ids = tokenizer(prompt,return_tensors="pt").input_ids.to(device)
                    outputs = model.generate(input_ids,max_length=400, min_length=300,do_sample=True, top_k=50, top_p=0.95) 
                    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace("\n","").replace("\t","").split(' ')
                    cur_len = len(prompt.split(' '))
                    text_all[item] = ' '.join(generated[cur_len:cur_len+100])
                    new_text.append(text_all[item])
                    del input_ids,outputs,generated
                model.train()

            text_all = np.array(text_all)
            text_emb_cur = obtain_text_emb(model,tokenizer,device,description,128,512,0)
            net = NeuralNetwork(512,384).to(device)
            criterion = nn.MSELoss()
            optimizer2 = optim.AdamW(net.parameters(), lr=0.001)
            emb_dataset = TensorDataset(text_emb_cur,text_emb_all)
            train_loader = DataLoader(emb_dataset, batch_size=32, shuffle=True)
            ones = torch.ones(32).to(device)
            # Train the neural network
            for ep in range(201):
                running_loss = 0.0
                for data in train_loader:
                    input_emb, output_emb = data
                    optimizer2.zero_grad()
                    predicted_output = net(input_emb)
                    loss = criterion(predicted_output, output_emb)
                    loss.backward()                                              
                    optimizer2.step()
                    running_loss += loss.item()
                if ep % 50 == 0:
                    print('Epoch %d loss: %.6f' % (ep + 1, running_loss / len(train_loader)))

            with torch.no_grad():
                text_emb_cur = net(obtain_text_emb(model,tokenizer,device,description[negatives],128,512,0))
                text_embedding = torch.zeros((num_item+1,384)).to(device)
                text_embedding[1:] = text_emb_all
                text_embedding[negatives_gpu] = text_emb_cur
                text_embedding = nn.Embedding.from_pretrained(text_embedding,freeze=False,padding_idx=0)
                predictions = full_sort_predict(MF_model,latest_interaction,text_embedding,text_embedding(negatives_gpu))
                maxv = torch.max(predictions,1)[0]
            
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=args.train_batch_size,
                collate_fn=data_collator,
            )

            results = bleu.compute(predictions=new_text, references=bleu_ref)
            bleu_score = results['bleu']
            results = meteor.compute(predictions=new_text, references=bleu_ref)
            meteor_score = results['meteor']
            avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len = evaluation(MF_model,preds,negatives,text_emb_all, text_all,device,num_user,num_item,sbert,test_dict,negatives_gpu)
            print('Epoch {}\t(target.rating, target.HITS, target.rank, text_similiarity, (bleu, meteor), avg_text_len)= {},{},{},{},({},{}),{}'.format(epoch,avg_rating,avg_HITS,avg_rank,avg_text_sim,bleu_score,meteor_score,avg_len),file=f,flush=True)
            if epoch==args.epochs:
                break
            avgs,act_avgs = [],[]
            print(new_text[0])
            original_loss,promotion_loss = 0,0
            text_embedding = nn.Embedding(num_item+1, 384 , padding_idx=0).to(device)
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                original_loss += loss.item()
           
                del inputs,outputs,labels
                torch.cuda.empty_cache()    

                cur_users = torch.from_numpy(np.random.choice(np.arange(num_user),size=int(0.1*num_user),replace=False)).long()
                cur_items = preds
                
                text_emb_cur = obtain_text_emb(model,tokenizer,device,description[cur_items],args.train_batch_size,512,1)
                text_emb_cur = net(text_emb_cur)

                cur_dict = copy.deepcopy(latest_dict)
                for key in cur_dict.keys():
                    cur_dict[key] = cur_dict[key][cur_users]
                cur_interaction = Interaction(cur_dict).to(device)

                cur_users = cur_users.to(device)
                cur_items = torch.from_numpy(cur_items).long().to(device)

                text_embedding = torch.zeros((num_item+1,384)).to(device)
                text_embedding[1:] = text_emb_all
                text_embedding[cur_items+1] = text_emb_cur
                
                text_embedding = nn.Embedding.from_pretrained(text_embedding,freeze=False,padding_idx=0)
                target_pred = full_sort_predict(MF_model,cur_interaction,text_embedding,text_emb_cur)
                avgs.append(torch.mean(target_pred).item())
                
                target_loss = loss2(target_pred,maxv[cur_users].unsqueeze(-1).expand(-1,len(preds)),indicator) 
                promotion_loss += args.coeff*target_loss.item()
                loss += args.coeff*target_loss
                loss.backward()
                optimizer.step()
                del target_pred,text_emb_cur
                torch.cuda.empty_cache()    

            print(original_loss,promotion_loss,np.average(avgs),file=f,flush=True)
            print("PROGRESS: {}%".format(round(100 * (epoch + 1) / args.epochs, 4)))
     
        np.savetxt('result/amazon/2FT/new_text_seed'+str(seed)+'_unisrec_opt350m.txt',text_all[preds],fmt='%s')
        del text_emb_all
        torch.cuda.empty_cache()            
        cur[0].append(avg_rating)
        cur[1].append(avg_HITS)
        cur[2].append(avg_rank)
        cur[3].append(avg_text_sim)
        cur[4].append(bleu_score)
        cur[5].append(meteor_score)
        cur[6].append(avg_len)

    print('original description result',file=f)
    print(orig,file=f)
    print('attack result',file=f)
    print(cur,file=f)


if __name__ == '__main__':
    main()
