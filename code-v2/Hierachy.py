import sys
import math
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import dense_to_sparse

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# import psutil
# import pynvml
# from ptflops.flops_counter import get_model_complexity_info

def gaussian_dis(x, sigma=0.1, epsilon=0.5):
    # n = x.shape[0]
    # w = np.ones([n, n]) - np.identity(n)
    return np.exp((-x ** 2) / (2 * sigma**2))


def get_initial_matrix(data_path):
    distance_matrix = np.load(f"{data_path}/ningxia_distance.npy")
    distance_matrix = gaussian_dis(distance_matrix)
    return torch.tensor(distance_matrix).type(torch.float)


def get_structure(k, grid_num, A_init, device):
    level_num =int(math.log(grid_num)/math.log(k))
    level_cluster_num = [int(grid_num*(1/k)**i) for i in range(1,level_num)] # 不包括初始层
    level_cluster_num.append(k)
    
    S_set = []
    RS_set = []
    A_set = []
    W_set = []
    Assign_set = [A_init]

    for l in range(level_num):
        assignment_vector = KMeans(n_clusters=level_cluster_num[l], random_state=0, n_init=10).fit(Assign_set[l]).labels_
        assignment_matrix_l =torch.from_numpy(OneHotEncoder(handle_unknown='ignore').fit_transform(assignment_vector.reshape(-1, 1)).toarray()).type(torch.float)
        adjcent_matrix_for_assignment_l = assignment_matrix_l.T @ Assign_set[l] @ assignment_matrix_l
        
        adjcent_matrix_for_assignment_l = F.softmax(adjcent_matrix_for_assignment_l,dim=-1)
        
        Assign_set.append(adjcent_matrix_for_assignment_l)
        S_set.append(assignment_matrix_l)

    level_cluster_num.insert(0,grid_num) # 包括初始层
    for l in range(level_num):
        adjcent_matrix_for_mp_l = np.zeros((level_cluster_num[l],level_cluster_num[l]))
        for i in range(adjcent_matrix_for_mp_l.shape[0]):
            ancestor_index_i = np.nonzero(S_set[l][i,:])[0].tolist()
            level_neighbor_index_i = np.nonzero(S_set[l][:,ancestor_index_i])[:,0].tolist()
            adjcent_matrix_for_mp_l[i,level_neighbor_index_i] = 1
        adjcent_matrix_for_mp_l_index, adjcent_matrix_for_mp_l_weights = dense_to_sparse(torch.from_numpy(adjcent_matrix_for_mp_l)*Assign_set[l])
        # weights没有经过norm
        A_set.append(adjcent_matrix_for_mp_l_index)
        W_set.append(adjcent_matrix_for_mp_l_weights)
    adjcent_matrix_for_mp_l_index, adjcent_matrix_for_mp_l_weights = dense_to_sparse(Assign_set[-1])
    A_set.append(adjcent_matrix_for_mp_l_index)
    W_set.append(adjcent_matrix_for_mp_l_weights)

    for l in range(level_num):    
        reverse_assignment_matrix_l = S_set[0]
        for s in range(1,l+1):
            reverse_assignment_matrix_l = reverse_assignment_matrix_l @ S_set[s]
        RS_set.append(reverse_assignment_matrix_l)

    S_set = [s.type(torch.long).to(device) for s in S_set]
    W_set = [s.type(torch.float).to(device) for s in W_set]
    A_set = [s.type(torch.long).to(device) for s in A_set]
    RS_set = [s.type(torch.long).to(device) for s in RS_set]
    
    return S_set, A_set, W_set, RS_set


# k = 3
# grid_num = 15
# A_init = torch.rand((grid_num,grid_num))
# S_set, A_set, W_set, RS_set = get_structure(k, grid_num, A_init,'cuda:0')
# print(233)