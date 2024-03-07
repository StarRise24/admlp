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

_dataset = "CARLA" # CARLA|NUSCENE
enable_tqdm = True
enable_image = True

def train_token():
    if _dataset == "NUSCENE":
        with open('fengze_nuscenes_infos_train.pkl','rb')as f:
            res=[]
            data=pickle.load(f)['infos']
            for ww in data:
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
        #print(res)
        for ww in res:
            pass
            #print(ww)
        return res
class TokenDataset(Dataset):
    def __init__(self,train=True):
        super(TokenDataset, self).__init__()
        self.train=train
        self.tokens=train_token() if train else test_token()

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
    model = VanillaPlanHead2(hidden_dim=512, dataset=_dataset, enable_image=enable_image)
    if enable_image:
        #optimizer = optim.AdamW(list(model.parameters()) + list(model.rgb_feature_extractor.parameters()),lr=4e-6,weight_decay=1e-2)
        optimizer = optim.AdamW(model.parameters(),lr=4e-6,weight_decay=1e-2)
    else:
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
    epoch_bar = trange(epochs, disable=not enable_tqdm)
    for epoch in epoch_bar:
        epoch_bar.set_description(f"Epoch progression")
        cnt=0
        total_loss = 0
        model.train()
        token_bar = tqdm(dataloader, leave=False, disable=not enable_tqdm)
        for token in token_bar:
            token_bar.set_description("Token progression")
            cnt+=len(token)

            optimizer.zero_grad()
            loss = model(token=token)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / cnt
        #writer.add_scalar('Loss/train', avg_loss, epoch)

        scheduler.step()
        #evaluate(model)
    torch.save(model.state_dict(), f'mlp_{_dataset}{"_imageenabled" if enable_image else ""}.pth')
    writer.close()


if __name__=='__main__':
    main()
    #print(train_token())