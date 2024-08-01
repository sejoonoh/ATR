#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2024-07-31
#Adversarial Text Rewriting for Text-aware Recommender Systems
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset,random_split
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from data.dataset import UniSRecDataset
from collections import defaultdict
from recbole.utils import FeatureSource, FeatureType
import torch.nn.utils.rnn as rnn_utils
from recbole.data.interaction import Interaction
import torch.nn as nn
import copy
from unisrec import UniSRec
from llama import Llama
from typing import List
import time

def full_sort_predict(MF_model, item_seq,item_seq_len,text_embedding,t_feat):
    with torch.autocast("cuda"): 
        item_emb_list = MF_model.moe_adaptor(text_embedding(item_seq))
        seq_output = MF_model.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = MF_model.moe_adaptor(t_feat)

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
    return scores

def fix_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

def evaluation(MF_model,target_items,negatives, orig_emb,cur_text,device,num_user,num_item,sbert,test_dict,cur_items):
    text_emb_all = sbert.encode(cur_text,convert_to_tensor=True,batch_size=1024,normalize_embeddings=True).to(device)
    avg_rank,avg_rating,avg_HITS = 0,0,0
    with torch.no_grad():
        text_embedding = torch.zeros((num_item+1,384)).to(device)
        text_embedding[1:] = text_emb_all
        text_embedding = nn.Embedding.from_pretrained(text_embedding,freeze=False,padding_idx=0)
        full_prediction = []
        item_id_list,item_length = test_dict['item_id_list'],test_dict['item_length']
        test_size = item_id_list.shape[0]
        for i in range((test_size-1)//4096+1):
            st_idx,ed_idx = i*4096,(i+1)*4096
            if ed_idx>test_size:
                ed_idx = test_size
            item_seq = item_id_list[st_idx:ed_idx]
            item_seq_len = item_length[st_idx:ed_idx]
            scores = full_sort_predict(MF_model,item_seq,item_seq_len,text_embedding, text_embedding(cur_items))
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


def ICL(dataset, pretrained_file, args, fix_enc=True):
    # configurations initialization
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    sbert = SentenceTransformer('all-MiniLM-L6-v2',device=device)
    f = open(args.output,'w') 
  
    props = ['src/props/UniSRec.yaml', 'src/props/finetune.yaml']

    # configurations initialization
    config = Config(model=UniSRec, dataset=dataset, config_file_list=props)
 
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = UniSRecDataset(config)
    logger.info(dataset)
    print(dataset.field2type)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    MF_model = UniSRec(config, train_data.dataset).to(device)

    X = pd.read_csv('src/dataset/downstream/amazon_book/amazon_rating.tsv',header=None,sep='\t').values   
    users,items = np.unique(X[:,0]),np.unique(X[:,1])
    num_users,num_items = len(users),len(items)
    user_map,item_map = {user:idx for (idx,user) in enumerate(users)},{item:idx for (idx,item) in enumerate(items)}
    for i in range(X.shape[0]):
        X[i,0] = user_map[X[i,0]]
        X[i,1] = item_map[X[i,1]]
 
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
 
    description = pd.read_csv('src/dataset/downstream/amazon_book/amazon_description_new.txt',header=None).values.flatten()
    text_emb_all = sbert.encode(description,convert_to_tensor=True,batch_size=1024,normalize_embeddings=True).to(device)
    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        MF_model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    MF_model.eval()
    # replace the below path with your own LLama path
    generator = Llama.build(
            ckpt_dir="llama/llama-2-7b-chat",
            tokenizer_path="llama/tokenizer.model",
            max_seq_len=2048,
            max_batch_size=4,
    )

    original_description = description
    print('loading done!')
    orig,cur = [[],[],[],[],[]],[[],[],[],[],[]]
    for seed in range(1):
        start_time = time.time()
        fix_random_seed(seed)
        target_items = pd.read_csv('result/amazon/target_item_seed'+str(seed)+'.txt',header=None).values.flatten()
        negatives = pd.read_csv('result/amazon/ranking_item_seed'+str(seed)+'.txt',header=None).values.flatten()
        negatives = negatives.astype(int)
        target_items = target_items.astype(int)
        negatives_gpu = torch.LongTensor(negatives).to(device)+1
        item_seq = test_dict['item_id_list']
        item_seq_len = test_dict['item_length']

        avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len = evaluation(MF_model, target_items,negatives,text_emb_all, original_description,device,num_users,num_items,sbert,test_dict,negatives_gpu)
        orig[0].append(avg_rating)
        orig[1].append(avg_HITS)
        orig[2].append(avg_rank)
        orig[3].append(avg_text_sim)
        orig[4].append(avg_len)

        cur_description = copy.deepcopy(original_description)
        cur_text = pd.read_csv('result/amazon/new_text_seed'+str(seed)+'_unisrec_opt350m.txt',header=None).values.flatten()
        
        prompts,target_idx = [],[]
        for i in range(len(cur_text)):
            sampled = np.random.choice([j for j in range(len(cur_text)) if j is not i],size=args.num_example,replace=False)
            instruction = "[INST] <<SYS>> You are an AI agent that helps a product seller on E-commerce. <</SYS>>\n"
 
            instruction += "Rewritten Examples:\n"
            for j in range(args.num_example):
                cur_item = sampled[j]
                instruction += cur_text[cur_item]+"\n"

            instruction += "Target product description:\n"+original_description[target_items[i]] + "\n"
            instruction += "Generate a new description for a target product based on its original description, so that the new description leads to the maximal rank of the target product across all users on the platform. The new description must follow the style of rewritten examples. The new description must be relevant to the original product description.\n"
            instruction += "Your answer (rewritten description only; no other text):\n[/INST]"

            prompts.append(instruction)
            target_idx.append(target_items[i])
            if len(prompts)==4:
                final_prompts: List[str] = prompts
                results = generator.text_completion(
                    final_prompts,
                    temperature=0.6,
                    top_p=0.9,
                )
                for (idx,result) in enumerate(results):
                    generation = result['generation'].replace('\n','')
                    cur_description[target_idx[idx]] = generation
                prompts = []
                target_idx = []
 
        for i in target_items:
            cur_str = cur_description[i].split(' ')
            if len(cur_str)>args.max_words:
                cur_description[i] = ' '.join(cur_str[:args.max_words])

        avg_rating,avg_HITS,avg_rank,avg_text_sim,avg_len = evaluation(MF_model, target_items,negatives,text_emb_all,cur_description,device,num_users,num_items,sbert,test_dict,negatives_gpu)
        np.savetxt('result/amazon/ICL/new_text_seed'+str(seed)+'_unisrec.txt',cur_description[target_items],fmt='%s')
        cur[0].append(avg_rating)
        cur[1].append(avg_HITS)
        cur[2].append(avg_rank)
        cur[3].append(avg_text_sim)
        cur[4].append(avg_len)
 
    print(time.time()-start_time,file=f,flush=True)
    print('Original description result',file=f,flush=True)
    print(orig,file=f,flush=True)
    print('Attack result',file=f,flush=True)
    print(cur,file=f,flush=True)
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ATR-ICL.")
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('-d', type=str, default='amazon_book', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path') 
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU number')
    parser.add_argument('--output', type=str, default='0',
                        help='GPU number')
    parser.add_argument('--emb_dim', type=int, default=384,
                        help='dimension of embedddings')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed number') 
    parser.add_argument('--num_example', type=int, default=10,
                        help='number of adversarial examples for ICL')  
    parser.add_argument('--max_words', type=int, default=100,
                    help='max number of words for rewriting')  

    args =  parser.parse_args()
      
    ICL(args.d, pretrained_file=args.p, args=args)

