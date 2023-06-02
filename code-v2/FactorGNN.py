import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv


class FactorGNN(nn.Module):
    def __init__(self, factor_num, dim_in, dim_out):
        super(FactorGNN, self).__init__()
        # self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        # self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.gcn = GCNConv(dim_in, dim_out, normalize=True)
        self.factorout = nn.Linear(factor_num*dim_in, dim_out)
        self.layernorm = nn.LayerNorm(10, eps=1e-12)
    
    def forward(self, x, factor_embeddings, grid_embeddings, supports):
        # x B,N,F,D
        # factor_embeddings F,e
        # grid_embeddings N,e
        # factor graphs N,F,F
        # output B,N,D
        B,N,F_,D = x.shape
        x = x.reshape(B,N*F_,D)
        grid_factor_embeddings = self.layernorm(factor_embeddings.unsqueeze(0)+grid_embeddings.unsqueeze(1)) # N,F,D
        factor_graphs = torch.einsum('nfd,ndk->nfk',grid_factor_embeddings, grid_factor_embeddings.permute(0,2,1)) # N,F,F
        factor_graphs = F.softmax(factor_graphs,dim=-1)
        factor_graphs = factor_graphs.reshape(N*F_,F_).unsqueeze(-2).repeat(1,N,1).reshape(N*F_,N*F_)
        factor_graphs = factor_graphs*supports
        # supports = torch.zeros((N*F_,N*F_)).to(x.device)
        # for i in range(N):
        #     supports[i*F_:(i+1)*F_,i*F_:(i+1)*F_] = F.softmax(factor_graphs[i,:,:],dim = -1)
        f_edge_index, f_edge_weights = dense_to_sparse(supports)
        x_gconv = F.relu(self.gcn(x, f_edge_index, f_edge_weights))
        
        x_gconv = x_gconv.reshape(B,N,F_,D).reshape(B,N,-1)
        x_out = self.factorout(x_gconv)
        
        return x_out