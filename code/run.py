import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from trainer import Trainer
from get_dataloader import *
from net import *
from util_l3 import *

def init_seed(seed):
    # torch.cuda.cudnn_enabled = False
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

model_name = 'WeatherGNN'
# parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--model', default='WeatherGNN', type=str, help='model name')
args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
args.add_argument('--log_dir', default=f"/home/wbq/nwp/ningxia_baselines/{model_name}/log{str(int(time.time()))}", type=str, help='log dir')
args.add_argument('--log_step', default=20, type=int)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=5, type=int)

args.add_argument('--debug', default=False, type=eval, help='debug mode')
args.add_argument('--seed', default=42, type=int, help='seed')
args.add_argument('--epochs', default=100, type=int)
args.add_argument('--batch_size', default=64, type=int, help='batch size')
args.add_argument('--grad_norm', default=False, type=eval)
args.add_argument('--max_grad_norm', default=5, type=int)
args.add_argument('--real_value', default=True, type=eval, help='true: 模型预测是scaled后的值')

args.add_argument('--mae_thresh', default=None, type=eval)
args.add_argument('--mape_thresh', default=0.1, type=float)

args.add_argument('--hidden_dim', default=16, type=int)
args.add_argument('--num_layers', default=3, type=int)
args.add_argument('--embed_dim', default=10, type=int)
args.add_argument('--time_len', default=7, type=int)
args.add_argument('--factor_num', default=11, type=int)
args.add_argument('--layer_1_node_num', default=31*41, type=int)
args.add_argument('--layer_2_node_num', default=128, type=int)
args.add_argument('--layer_3_node_num', default=32, type=int)

args = args.parse_args()
init_seed(args.seed)

# A_1 = np.random.rand(args.layer_1_node_num,args.layer_1_node_num)
A_1 = get_matrix('/mnt/workspace/NWP_data/data/ningxia_matrix.npy')
S_set, A_sets, neighbor_index_layer_2, neighbor_index_layer_3 = get_structure(A_1, args.layer_1_node_num, args.layer_2_node_num, args.layer_3_node_num)

args.s1 = torch.Tensor(S_set[0]).to(args.device)
args.s2 = torch.Tensor(S_set[1]).to(args.device)

args.a1 = F.softmax(torch.Tensor(A_sets[0]),dim=1).to(args.device)
args.a2 = F.softmax(torch.Tensor(A_sets[1]),dim=1).to(args.device)
args.a3 = F.softmax(torch.Tensor(A_sets[2]),dim=1).to(args.device)

args.neighbor_index_layer_2 = neighbor_index_layer_2
args.neighbor_index_layer_3 = neighbor_index_layer_3

model = WeatherGNN(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args.batch_size)

loss = torch.nn.L1Loss().to(args.device)
lr_scheduler = None
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.003, eps=1.0e-8, weight_decay=0, amsgrad=False)

trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler=lr_scheduler)
trainer.train()

# print(233)