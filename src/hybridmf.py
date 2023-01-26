import os
import time
import random
import argparse
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, Dataset,random_split
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')
import textattack
import transformers
import evaluate

class GPTDataset(Dataset):
        def __init__(self, txt_list, tokenizer, max_length):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []                    
            for txt in txt_list:            
                encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>',truncation=True, max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                                                                                      
        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx]

def fix_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

class HMF(nn.Module):
    def __init__(self, ratings,emb_dim):
        super(HMF, self).__init__()
        self.num_users = len(np.unique(ratings[:,0]))
        self.num_items = len(np.unique(ratings[:,1]))
        self.emb_dim = emb_dim
        self.user_layers_sizes = [self.num_items,1024,1024,self.emb_dim]
        self.item_layers_sizes = [self.num_users,1024,1024,self.emb_dim]
        self.dropout_ratio = [0.2,0.1,0.3]
        self.weight = nn.Parameter(torch.tensor([1.0,1.0], requires_grad=True))
        self.fc_users,self.fc_items = nn.ModuleList(),nn.ModuleList()
        for (idx,(in_size, out_size)) in enumerate(zip(self.user_layers_sizes[:-1], self.user_layers_sizes[1:])):
            self.fc_users.append(nn.Linear(in_size, out_size))
            self.fc_users.append(nn.ReLU())
            self.fc_users.append(nn.Dropout(p=self.dropout_ratio[idx]))
        for (idx,(in_size, out_size)) in enumerate(zip(self.item_layers_sizes[:-1], self.item_layers_sizes[1:])):
            self.fc_items.append(nn.Linear(in_size, out_size))
            self.fc_items.append(nn.ReLU())
            self.fc_items.append(nn.Dropout(p=self.dropout_ratio[idx]))

    def forward(self,user_embs,item_embs,text_embs):
        for i in range(9):
            user_embs = self.fc_users[i](user_embs)
            item_embs = self.fc_items[i](item_embs)
       
        probs = F.softmax(self.weight) 
        preds = nn.ReLU()(torch.sum(user_embs*(probs[0]*item_embs+probs[1]*text_embs),dim=-1))
        return preds

class MF_dataset(Dataset):
    def __init__(self,data,text_emb,emb_dim):
        self.data = data
        self.text_emb = text_emb
        self.num_samples = data.shape[0]

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, item):
        return (int(self.data[item,0]),int(self.data[item,1]),self.text_emb[item],self.data[item,2])

def evaluation_MF(MF_model,target_items,negatives, orig_emb,cur_text,test,device,rating_matrix):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run HybridMF.")
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU number')
    parser.add_argument('--output', type=str, default='0',
                        help='GPU number')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of training data')
    parser.add_argument('--emb_dim', type=int, default=384,
                        help='dimension of embedddings')
    parser.add_argument('--seed', type=int, default=0,
                        help='Dimension of embedddings') 
    parser.add_argument('--mode', type=int, default=0,
                        help='dimension of embedddings')  
    parser.add_argument('--max_words', type=int, default=100,
                    help='dimension of embedddings')  

    args =  parser.parse_args()
    fix_random_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X = pd.read_csv('data/movielens_rating.tsv',header=None,sep='\t').values   
    users,items = np.unique(X[:,0]),np.unique(X[:,1])
    user_map,item_map = {user:idx for (idx,user) in enumerate(users)},{item:idx for (idx,item) in enumerate(items)}
    rating_matrix = torch.zeros((len(users),len(items)))
    i1,i2,i3 = [],[],[]
    for i in range(X.shape[0]):
        X[i,0] = user_map[X[i,0]]
        X[i,1] = item_map[X[i,1]]
        if np.random.random()<=args.train_ratio:
            if np.random.random()<=0.9:
                i1.append(i)
            else:
                i2.append(i)
            rating_matrix[int(X[i,0]),int(X[i,1])]=X[i,2]
        else:
            i3.append(i)
    train,val,test = X[i1],X[i2],X[i3]
    minv,maxv = min(X[:,2]),max(X[:,2])
    description = pd.read_csv('data/movielens_description.txt',header=None).values.flatten()
    text_emb_all = sbert.encode(description,convert_to_tensor=True,batch_size=4096)
    MF_model = HMF(X,args.emb_dim).to(device)
    text_emb = text_emb_all[train[:,1].astype(int),:]
    train_data = MF_dataset(train,text_emb,args.emb_dim)
    text_emb = text_emb_all[val[:,1].astype(int),:]
    val_data = MF_dataset(val,text_emb,args.emb_dim)
    text_emb = text_emb_all[test[:,1].astype(int),:]
    test_data = MF_dataset(test,text_emb,args.emb_dim)

    batch_size = args.batch_size
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_data,shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_data,shuffle=True, batch_size=batch_size)

    bleu = evaluate.load("bleu")
    meteor = evaluate.load('meteor')
    bertscore = evaluate.load("bertscore")
 
    if abs(args.mode)==5:
        model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        avg_ratings = [[] for i in range(len(items))]
        avg = 0
        for i in range(X.shape[0]):
            avg_ratings[int(X[i,1])].append(X[i,2])
            avg+=float(X[i,2])
        avg/=X.shape[0]

#        attack = textattack.attack_recipes.bert_attack_li_2020.BERTAttackLi2020.build(model_wrapper)
#        attack = textattack.attack_recipes.checklist_ribeiro_2020.CheckList2020.build(model_wrapper)
        attack = textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build(model_wrapper)
        train_pairs,test_pairs = [],[]
        for i in range(len(items)):
            label = 0 if np.average(avg_ratings[i])<avg else 1
            if np.random.random()<=0.9:
                train_pairs.append((description[i],label))
            else:
                test_pairs.append((description[i],label))

        train_dataset,test_dataset = textattack.datasets.Dataset(train_pairs), textattack.datasets.Dataset(test_pairs)
        training_args = textattack.TrainingArgs(num_epochs=10,num_clean_epochs=10,num_train_adv_examples=1,learning_rate=5e-5,per_device_train_batch_size=16)
        trainer = textattack.Trainer(model_wrapper,"classification",attack,train_dataset,test_dataset,training_args)
        trainer.train()
    
    if args.mode==0:
        optimizer = optim.AdamW(MF_model.parameters(), lr=0.001)
        MSE = nn.MSELoss()
        best_model,best_RMSE = 0,999999
        for epoch in range(args.epochs):
            total_loss = 0
            MF_model.train()
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                user_emb,item_emb = rating_matrix[batch[0],:].to(device),rating_matrix[:,batch[1]].t().to(device)
                predictions = MF_model(user_emb,item_emb,batch[2].to(device))
                loss = MSE(predictions, batch[3].float().to(device))
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            if epoch%1==0:
                print('train_loss and weight = {}\t{}'.format(total_loss/train.shape[0],F.softmax(MF_model.weight)))
                val_RMSE,test_RMSE = 0,0
                MF_model.eval()
                with torch.no_grad():
                    for step, batch in enumerate(val_dataloader):
                        user_emb,item_emb = rating_matrix[batch[0],:].to(device),rating_matrix[:,batch[1]].t().to(device)
                        predictions = MF_model(user_emb,item_emb,batch[2].to(device))
                        predictions[predictions<minv] = minv
                        predictions[predictions>maxv] = maxv
                        val_RMSE += torch.sum((predictions-batch[3].float().to(device))**2)
 
                    for step, batch in enumerate(test_dataloader):
                        user_emb,item_emb = rating_matrix[batch[0],:].to(device),rating_matrix[:,batch[1]].t().to(device)
                        predictions = MF_model(user_emb,item_emb,batch[2].to(device))
                        predictions[predictions<minv] = minv
                        predictions[predictions>maxv] = maxv
                        test_RMSE += torch.sum((predictions-batch[3].float().to(device))**2)
                val_RMSE=torch.sqrt(val_RMSE/val.shape[0])
                test_RMSE=torch.sqrt(test_RMSE/test.shape[0])
                print('(val,test RMSE = {},{}) @ Epoch {}'.format(val_RMSE.item(),test_RMSE.item(),epoch+1))
                if best_RMSE>val_RMSE:
                    best_RMSE=val_RMSE
                    best_model = copy.deepcopy(MF_model)
        torch.save(best_model.state_dict(), 'ckpt/movielens_HMF')
        best_model.eval()
        with torch.no_grad():
            new_train = []
            for step, batch in enumerate(train_dataloader):
                cur_user,cur_item = batch[0],batch[1]
                user_emb,item_emb = rating_matrix[cur_user,:].to(device),rating_matrix[:,cur_item].t().to(device)
                predictions = best_model(user_emb,item_emb,batch[2].to(device)).detach().cpu().numpy()
                for i in range(len(predictions)):
                    new_train.append([users[cur_user[i]],items[cur_item[i]],predictions[i]])
            for i in range(val.shape[0]):
                val[i,0] = users[int(val[i,0])]
                val[i,1] = items[int(val[i,1])] 
            for i in range(test.shape[0]):
                test[i,0] = users[int(test[i,0])]
                test[i,1] = items[int(test[i,1])]
            np.savetxt('data/movielens_HMF_train.tsv',np.array(new_train),fmt='%s\t%s\t%s')
            np.savetxt('data/movielens_HMF_val.tsv',val,fmt='%s\t%s\t%s') 
            np.savetxt('data/movielens_HMF_test.tsv',test,fmt='%s\t%s\t%s')

    else:
        with torch.no_grad():
            MF_model.load_state_dict(torch.load('ckpt/movielens_HMF'))
            MF_model.eval()
            rating_matrix = rating_matrix.to(device)
            print(F.softmax(MF_model.weight))
            original_description = description
            keywords = pd.read_csv('data/movielens_description.key.txt',header=None).values.flatten()
            all_words,all_keywords = [],[]
            for i in range(description.shape[0]):
                all_words += description[i].split(' ')
            all_words = np.unique(all_words)
            print(all_words[:10],len(all_words))
            for item in range(len(items)):
                all_keywords += keywords[item].split(' ')
            all_keywords = np.unique(all_keywords)
            print(all_keywords[:10],len(all_keywords))           
        print('loading done!')
        orig,cur = [[],[],[],[],[]],[[],[],[],[],[],[],[],[]]
        for seed in range(5):
            fix_random_seed(seed)
            target_items = pd.read_csv('result/movielens/target_item_seed'+str(seed)+'.txt',header=None).values.flatten()
            negatives = pd.read_csv('result/movielens/ranking_item_seed'+str(seed)+'.txt',header=None).values.flatten()
            negatives = negatives.astype(int)
            target_items = target_items.astype(int)

            if args.mode<0:
                original_description = copy.deepcopy(description)
                for item in target_items:
                    new_str = ' '.join(description[item].split(' ')[:10])
                    original_description[item] = new_str

            text_emb_all = sbert.encode(original_description,convert_to_tensor=True,batch_size=4096).to(device)
            if abs(args.mode)<=2:
                pointer_description = pd.read_csv('result/movielens/POINTER_seed'+str(seed)+'.txt',header=None).values.flatten()
                new_description = pd.read_csv('result/movielens/new_text_seed'+str(seed)+'_HMF_ft10ep5.txt',header=None).values.flatten()
            random_description = copy.deepcopy(original_description)
            keyword_description = copy.deepcopy(original_description)
            for item in target_items:
                current_sentence = np.array(original_description[item].split(' '))
                chosen_indices = np.random.choice(np.arange(len(current_sentence)),size=int(0.1*len(current_sentence)))
                #new_sentence = np.insert(current_sentence,chosen_indices,np.random.choice(all_words,size=len(chosen_indices)))
                current_sentence[chosen_indices] = np.random.choice(all_words,size=len(chosen_indices))
                random_description[item] = ' '.join(list(current_sentence))
            
                current_sentence = np.array(original_description[item].split(' '))
                chosen_indices = np.random.choice(np.arange(len(current_sentence)),size=int(0.1*len(current_sentence)))
                current_sentence[chosen_indices] = np.random.choice(all_keywords,size=len(chosen_indices))
                keyword_description[item] = ' '.join(list(current_sentence))
                
            avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len = evaluation_MF(MF_model, target_items,negatives,text_emb_all, original_description,test,device,rating_matrix)
            orig[0].append(avg_rating)
            orig[1].append(avg_HITS)
            orig[2].append(avg_rank)
            orig[3].append(avg_text_sim)
            orig[4].append(avg_len)
            if args.mode==1:
                cur_description = new_description
            if args.mode==2:
                cur_description = pointer_description
            if args.mode==3:
                cur_description = random_description
            if args.mode==4:
                cur_description = keyword_description
            if abs(args.mode)==5:
                cur_data = []
                for item in target_items:
                    label = 0 if np.average(avg_ratings[item])<avg else 1
                    cur_data.append((original_description[item],label))
                cur_dataset = textattack.datasets.Dataset(cur_data)
                torch.cuda.empty_cache()
                #attack = textattack.attack_recipes.clare_li_2020.CLARE2020.build(model_wrapper)
                #attack = textattack.attack_recipes.hotflip_ebrahimi_2017.HotFlipEbrahimi2017.build(model_wrapper)
                #attack = textattack.attack_recipes.bae_garg_2019.BAEGarg2019.build(model_wrapper)
                #attack = textattack.attack_recipes.checklist_ribeiro_2020.CheckList2020.build(model_wrapper)
                attack = textattack.attack_recipes.a2t_yoo_2021.A2TYoo2021.build(model_wrapper)
                attack_args = textattack.AttackArgs(num_examples=-1, log_to_csv="log4.csv",random_seed=seed,disable_stdout=True)
                attacker = textattack.Attacker(attack, cur_dataset, attack_args)
                attack_results = attacker.attack_dataset()
                cur_description = copy.deepcopy(original_description)
                perturbed = pd.read_csv('log4.csv',sep=',',header=0)['perturbed_text']
                for i in range(len(target_items)):
                    new_text = perturbed[i].strip('[').strip(']')
                    cur_description[target_items[i]] = new_text 
            if abs(args.mode)>=6:
                tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
                model = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(device)
                cur_description = copy.deepcopy(original_description)
                if abs(args.mode)==7:
                    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2',bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')
                    model.resize_token_embeddings(len(tokenizer))
                    dataset = GPTDataset(list(description), tokenizer, max_length=256)
                    train_size = int(0.9 * len(dataset))
                    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
                    training_args = transformers.TrainingArguments(output_dir='outputs/', num_train_epochs=10, save_strategy = 'no', per_device_train_batch_size=32, per_device_eval_batch_size=32,logging_strategy='no', report_to = 'none')
                    transformers.Trainer(model=model,args=training_args,train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),'attention_mask': torch.stack([f[1] for f in data]),'labels': torch.stack([f[0] for f in data])}).train()
                    model.save_pretrained('ckpt/GPT_FT10_ML')
                    tokenizer.save_pretrained('ckpt/GPT_FT10_ML')
#                    tokenizer = transformers.GPT2Tokenizer.from_pretrained('ckpt/GPT_FT10')
#                    model = transformers.GPT2LMHeadModel.from_pretrained('ckpt/GPT_FT10').to(device) 

                for (idx,item) in enumerate(target_items):
                    input_ids = tokenizer('<|startoftext|> '+cur_description[item], return_tensors="pt").input_ids.to(device)
                    outputs = model.generate(input_ids, max_new_tokens = 250, min_length=300,do_sample=True,top_k=50,top_p=0.95)
                    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split(' ')
                    cur_description[item] = ' '.join(generated[100:]) if args.mode>0 else ' '.join(generated[10:])
                    cur_description[item] = cur_description[item].replace("\n", "")
                print(cur_description[target_items[0]])


            bleu_ref,new_text = [],[]
            for i in target_items:
                cur_str = cur_description[i].split(' ')
                if len(cur_str)>args.max_words:
                    cur_description[i] = ' '.join(cur_str[:args.max_words])
                
                bleu_ref.append(description[i])
                new_text.append(cur_description[i])

            results = bleu.compute(predictions=new_text, references=bleu_ref)
            bleu_score = results['bleu']
            results = meteor.compute(predictions=new_text, references=bleu_ref)
            meteor_score = results['meteor']
            results = bertscore.compute(predictions=new_text, references=bleu_ref, lang='en')
            bert_score = np.average(results['f1'])
 
            torch.cuda.empty_cache()
            avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len = evaluation_MF(MF_model, target_items,negatives,text_emb_all, cur_description,test,device,rating_matrix)
            cur[0].append(avg_rating)
            cur[1].append(avg_HITS)
            cur[2].append(avg_rank)
            cur[3].append(avg_text_sim)
            cur[4].append(bleu_score)
            cur[5].append(meteor_score)
            cur[6].append(bert_score)
            cur[7].append(avg_len)
 
        f = open(args.output,'w') 
        print('Original description result',file=f)
        print(orig,file=f)
        print('Attack result',file=f)
        print(cur,file=f)
