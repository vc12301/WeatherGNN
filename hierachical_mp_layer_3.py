import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util_l3 import *
from net import *


# 假设有3层
batch_size = 64
hidden_dim = 64

num_layers = 3
layer_1_node_num = 58*47
layer_2_node_num = 128
layer_3_node_num = 32

device = 'cuda:0'

A_1 = np.random.rand(layer_1_node_num,layer_1_node_num)
# A_1 = get_distance_matrix(path)
S_set, A_sets, neighbor_index_layer_2, neighbor_index_layer_3 = get_structure(A_1, layer_1_node_num, layer_2_node_num, layer_3_node_num)

A_1 = torch.Tensor(A_1).to(device)
S_set = [torch.Tensor(S_set[i]).to(device) for i in range (num_layers-1)]
A_sets = [F.softmax(torch.Tensor(A_sets[i]),dim=1).to(device) for i in range (num_layers-1)]

# 动态的输入特征
x = torch.rand(batch_size,layer_1_node_num,hidden_dim).to(device)
fhmp = FHMP(hidden_dim,num_layers).to(device)
x_updated = fhmp(x, S_set, A_sets, neighbor_index_layer_2, neighbor_index_layer_3)

print(x_updated.shape)
    