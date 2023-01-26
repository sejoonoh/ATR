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
    
    X = pd.read_csv('data/amazon_rating.tsv',header=None,sep='\t').values   
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
    description = pd.read_csv('data/amazon_description.txt',header=None).values.flatten()
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
        torch.save(best_model.state_dict(), 'ckpt/amazon_HMF')
