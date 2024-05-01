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
from time import time
import re
from evaluate_for_mlp import run

_dataset = "CARLA" # CARLA|NUSCENE
enable_tqdm = True
enable_image = False

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
            res = data
            return res  

def test_token():
    if _dataset == "NUSCENE":
        with open('stp3_val/filter_token.pkl','rb')as f:
            res=pickle.load(f)
            return res
    elif _dataset == "CARLA":
        with open('validation.pkl','rb')as f:
            res=pickle.load(f)
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

def evaluate(model, dataset_type, writer, epoch):
    dataset = TokenDataset(train=False)
    res = {}
    for i in trange(len(dataset)):
        token = [dataset[i]]
        pred = model.inference(token=token)
        res[token[0]] = pred
    with open('output_data.pkl','wb')as f:
        pickle.dump(res,file=f)
    run(dataset_type, writer, epoch)


def main():
    writer = SummaryWriter(f'runs/{_dataset}/train27')
    model = VanillaPlanHead2(hidden_dim=512, dataset=_dataset, enable_image=enable_image)
    optimizer = optim.AdamW(model.parameters(),lr=4e-6,weight_decay=1e-2)
    batch_size = 4
    dataset = TokenDataset() 
    dataloader = DataLoader(dataset,batch_size,shuffle=True)
    device = torch.device('cuda:0')
    model = model.to(device)

    evaluate(model, _dataset, writer, 0)

    epochs = 5
    if enable_image:
        epochs = 30
    scheduler = MultiStepLR(optimizer,[2,4],gamma=0.2)
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
        writer.add_scalar('Loss/train', avg_loss, epoch+1)

        scheduler.step()
        evaluate(model, _dataset, writer, epoch+1)
        #carla_l2_eval(model, writer, epoch)
    writer.flush()
    torch.save(model.state_dict(), f'mlp_{_dataset}{"_imageenabled" if enable_image else ""}_{time()}.pth')
    writer.close()


if __name__=='__main__':
    main()
    #gt_occup = open('stp3_val/stp3_occupancy.pkl','rb')
    #gt_traj_occup = pickle.load(gt_occup)
    #print(gt_traj_occup.keys())
    #print(gt_traj_occup['c5f58c19249d4137ae063b0e9ecd8b8e'].shape)

    gt_traj = open('stp3_val/stp3_traj_gt.pkl','rb')
    gt_traj_traj = pickle.load(gt_traj)
    print(gt_traj_traj['c5f58c19249d4137ae063b0e9ecd8b8e'])
    #print(train_token())