import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from FactorGNN import FactorGNN
from FHGNN import *


class WeatherGNN(nn.Module):
    def __init__(self, args):
        super(WeatherGNN, self).__init__()
        self.level_num = args.level_num
        self.factor_num = args.factor_num
        self.grid_num = args.grid_num
        self.out_num = args.out_num
        
        self.time_len = args.time_len
        
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        
        self.factor_embeddings = nn.Parameter(torch.randn(self.factor_num, self.embed_dim), requires_grad=True)
        self.grid_embeddings = nn.Parameter(torch.randn(self.grid_num, self.embed_dim), requires_grad=True)

        self.input_embed = nn.Linear(self.time_len,self.hidden_dim)
        self.factorgnn =  FactorGNN(self.factor_num,self.hidden_dim,self.hidden_dim)
        self.fhgnn = FHGNN(self.level_num, self.hidden_dim)

        # self.decoder = nn.Linear(self.hidden_dim, self.out_num)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.out_num),
            nn.Dropout(0.1)
        )
        
        
    def forward(self, x, assignment_list, reversed_assignment_list, edge_index_list, edge_weight_list, supports):
        B,T,N,F_ = x.shape
        x = self.input_embed(x.permute(0,2,3,1)) # B,N,F,D
        factorgnn_out = self.factorgnn(x,self.factor_embeddings,self.grid_embeddings, supports)  # B,N,D
        fhgnn_out = self.fhgnn(factorgnn_out,assignment_list, reversed_assignment_list, edge_index_list, edge_weight_list) # B,N,D
        out = self.decoder(fhgnn_out)
        # out = F.softmax(out,dim=1)
        # x = x.permute(0, 2, 1)
        # x = F.relu(self.decoder(x))
        # x = x.permute(0, 2, 1) 

        return out
