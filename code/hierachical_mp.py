import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import psutil
import pynvml
from ptflops.flops_counter import get_model_complexity_info

# from weathergnn import WeatherGNN
# from hierachical_mp import *
from util_l3 import *
from net import *

device = 'cuda:0'


pynvml.nvmlInit()
# model = MLP(args.input_dim,2,dropout=0.)
# model = BiLSTM(args.input_dim,args.hidden_dim,args.output_dim)
# model = ConvLSTM(args.input_dim, args.hidden_dim, (args.kernel_size,args.kernel_size), args.num_layers, True, True, True)
res = {
    'Flops': [],
    'Params': [],
    'GPU':[]
}

args = argparse.ArgumentParser(description='arguments')
args.add_argument('--batch_size', default=64, type=int)
args.add_argument('--hidden_dim', default=16, type=int)
args.add_argument('--num_layers', default=3, type=int)
args.add_argument('--embed_dim', default=10, type=int)
args.add_argument('--time_len', default=7, type=int)
args.add_argument('--factor_num', default=11, type=int)
# args.add_argument('--layer_1_node_num', default=2500, type=int)
args.add_argument('--layer_2_node_num', default=128, type=int)
args.add_argument('--layer_3_node_num', default=32, type=int)
args = args.parse_args()


# try:
# for h_ in range(50,1001,50):
h_ = 50
cal_size = h_*h_

args.layer_1_node_num = cal_size

A_1 = np.random.rand(args.layer_1_node_num,args.layer_1_node_num)
# A_1 = get_distance_matrix(path)
S_set, A_sets, neighbor_index_layer_2, neighbor_index_layer_3 = get_structure(A_1, args.layer_1_node_num, args.layer_2_node_num, args.layer_3_node_num)

args.s1 = torch.Tensor(S_set[0]).to(device)
args.s2 = torch.Tensor(S_set[1]).to(device)

args.a1 = F.softmax(torch.Tensor(A_sets[0]),dim=1).to(device)
args.a2 = F.softmax(torch.Tensor(A_sets[1]),dim=1).to(device)
args.a3 = F.softmax(torch.Tensor(A_sets[1]),dim=1).to(device)

args.neighbor_index_layer_2 = neighbor_index_layer_2
args.neighbor_index_layer_3 = neighbor_index_layer_3

# np.save('S_set.npy',S_set)
# np.save('A_sets.npy',A_sets)
# np.save('neighbor_index_layer_2.npy',neighbor_index_layer_2)
# np.save('neighbor_index_layer_3.npy',neighbor_index_layer_3)

# x = torch.rand(1,7,args.layer_1_node_num,args.factor_num).to(device)
model = WeatherGNN(args).to(device)
# x_updated = model(x)

with torch.cuda.device(0):
    flops, params = get_model_complexity_info(model, (7,h_*h_,11), as_strings=True, print_per_layer_stat=True)
    # gpu_memory = psutil.virtual_memory().total - psutil.virtual_memory().available
    # gpu = gpu_memory / 1024 / 1024 /1024 #GB
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    res['Flops'].append(flops)
    res['Params'].append(params)
    res['GPU'].append(mem_info.used/1024**2)
    print('Flops:  ' + flops)


# print(x_updated.shape)