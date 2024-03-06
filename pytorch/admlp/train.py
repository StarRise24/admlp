import os

from planner import VanillaPlanHead2
import torch
from torch import nn
from torch.nn import functional as F
import pickle
import numpy as np
import math
from torchmetrics import Metric
import copy
import math
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm,trange
import re
#from evaluate_for_mlp import run

_dataset = "NUSCENE" # CARLA|NUSCENE

def train_token():
    if _dataset == "NUSCENE":
        with open('fengze_nuscenes_infos_train.pkl','rb')as f:
            res=[]
            data=pickle.load(f)['infos']
            #print(data)
            #with open("dump.txt", "w") as dump:
            #    regex = r"[a-zA-Z_]+\d?[a-zA-Z_]+"
            #    dump.write(str(list(re.split("[a-zA-Z_]+\d?[a-zA-Z_]+", str(data)))))
            #exit()
            for ww in data:
                #print(ww)
                #exit()
                res.append(ww['token'])
            return res
    elif _dataset == "CARLA":
        with open("./train.pkl", 'rb') as f:
            data = pickle.load(f)
            res = list(range(len(data)))
            return res  

def test_token():
    with open('stp3_val/filter_token.pkl','rb')as f:
        res=pickle.load(f)
        return res

def test_token2():
    with open('stp3_val/data_nuscene.pkl','rb')as f:
        res=pickle.load(f)
        print(res)
        for ww in res:
            print(ww)
        return res
class TokenDataset(Dataset):
    def __init__(self,train=True):
        super(TokenDataset, self).__init__()
        self.train=train
        self.tokens=train_token() if train else test_token2()

    def __getitem__(self, item):
        return self.tokens[item]

    def __len__(self):
        return len(self.tokens)

def evaluate(model):
    dataset = TokenDataset(train=False)
    res = {}
    for i in trange(len(dataset)):
        token = [dataset[i]]
        pred = model.inference(token=token)
        res[token[0]] = pred
    with open('output_data.pkl','wb')as f:
        pickle.dump(res,file=f)
    #run()

class CarlaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        return features, label

def main():
    with open('train.pkl', 'rb') as f:
        data = pickle.load(f)

    writer = SummaryWriter('runs/train26')
    model = VanillaPlanHead2(hidden_dim=512, dataset=_dataset)
    optimizer = optim.AdamW(model.parameters(),lr=4e-6,weight_decay=1e-2)
    batch_size = 4
    dataset = TokenDataset()
    #dataset = CarlaDataset(data)
    dataloader = DataLoader(dataset,batch_size,shuffle=True)
    device = torch.device('cuda:0')
    model = model.to(device)
    epochs = 6
    scheduler = MultiStepLR(optimizer,[2,4],gamma=0.2)
    # evaluate(model)
    for epoch in trange(epochs):
        cnt=0
        total_loss = 0
        model.train()
        for token in dataloader:
            cnt+=len(token)

            optimizer.zero_grad()
            loss = model(token=token)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / cnt
        #writer.add_scalar('Loss/train', avg_loss, epoch)

        scheduler.step()
        evaluate(model)
    torch.save(model.state_dict(), 'mlp26MBdata.pth')
    writer.close()


if __name__=='__main__':
    main()
    #print(train_token())