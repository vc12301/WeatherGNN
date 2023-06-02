import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class FHGNN(nn.Module):
    def __init__(self, num_levels, dim):
        super().__init__()
        self.num_levels = num_levels
        self.dim = dim
        self.input_embed = nn.Linear(self.dim,self.dim)
        self.att_q = nn.Linear(self.dim, self.dim)
        self.att_k = nn.Linear(self.dim, self.dim)
        self.gnn_layers = nn.ModuleList([FHGNNLayer(dim=self.dim) for _ in range(self.num_levels)])
        
        
    def forward(self, x, assignment_list, reversed_assignment_list, edge_index_list, edge_weight_list):
        # x: [B, N, d]
        # assignment_list: the list of assginments of each level, where the l-th assignment is an N^(l) dimensional vector (the sparse format of assginment matrix) (L-1)
        # reversed_assignment_list: the list of reversed assginments of each level (l>0) to level 0, where each reversed assginment is an N-d vector (L-1)
        # edge_index_list: the list of edge index of each level, where the l-the edge index: [E, 2], corresponding to the graph of l-th level (L)
        # edge_weight_list: the corresponding predefined edge weights, where the l-the edge weight: [E, 1] (L)
        
        h = self.input_embed(x)        
        h_q = self.att_q(x)
        h_k = self.att_k(x)
        
        h_prime = x + h
        h_prime += self.gnn_layers[0](h, h_q, h_k, edge_index_list[0], edge_weight_list[0])
        for i in range(self.num_levels):
            high_grid_num = assignment_list[i].shape[1]
            low_index, high_index = torch.nonzero(assignment_list[i]).T
            _, dup_index = torch.nonzero(reversed_assignment_list[i]).T
            
            h = self.get_high(h, high_grid_num, high_index)
            # h * w might be here 
            h_q = self.get_high(h_q, high_grid_num, high_index)
            h_k = self.get_high(h_k, high_grid_num, high_index)
            h_hat = self.gnn_layers[0](h, h_q, h_k, edge_index_list[i+1], edge_weight_list[i+1])
            h_prime += self.duplication(h_hat, dup_index)
            
        return h_prime

    def get_high(self, h, n_h, h_index):
        # should not be implemented with matrix multiplication, but with non-parametric message passing
        B,n_l,D = h.shape
        h_high = torch.zeros((B,n_h,D)).cuda()
        h_high = h_high.index_add_(dim=1, index=h_index, source=h)
        return h_high
    
    def duplication(self, h, dup_index):
        # should not be implemented with matrix multiplication, but with indexing operation
        h_0 = h[:,dup_index,:]
        return h_0
        

class FHGNNLayer(MessagePassing):
    def __init__(self, dim):
        super(FHGNNLayer, self).__init__(aggr='add')
        self.dim = dim
        self.lin_out = nn.Linear(self.dim, self.dim)

    def forward(self, x, h_q, h_k, edge_index, edge_attr):
        B,N,D = x.shape
        _,E = edge_index.shape

        edge_index_q = edge_index[0,...]
        edge_index_k = edge_index[1,...]
        
        h_q_gather = h_q[:,edge_index_q,:] # [B, E, d]
        h_k_gather = h_k[:,edge_index_k,:] # [B, E, d]
        
        batch_edge_attr = edge_attr.unsqueeze(0).repeat(B,1).unsqueeze(-1) # B,E,1
        
        # dynamic adjustment
        sim = torch.einsum('bed,bed->be', h_q_gather, h_k_gather).unsqueeze(2) / torch.sqrt(torch.tensor(self.dim))
        batch_edge_attr *= sim
        batch_edge_attr = F.softmax(batch_edge_attr, dim=1)
        
        # calculate softmax 
        # 这一步有问题 容易产生nan
        # 为什么 因为batch_edge_attr中有inf
        # 这个inf是由sim计算exp造成的 sim的max会有600多 可以先对边进行了一个softmax
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: 
        # [torch.cuda.FloatTensor [64, 9, 1]], which is output 0 of DivBackward0, 
        # is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. 
        # The variable in question was changed in there or anywhere later. Good luck!
        # batch_edge_attr = batch_edge_attr.exp().type(torch.float) # B,E,1
        # norm_factor = torch.zeros((B,N,1)).cuda()
        # norm_factor = norm_factor.index_add_(dim=1, index=edge_index_q, source=batch_edge_attr) #(B,N,1)
        # norm_factor_gather = norm_factor[:,edge_index_q,:] #(B,E,1)
        # batch_edge_attr /= norm_factor_gather #(B,E,1)
        
        return self.propagate(edge_index, x=x, edge_weight=batch_edge_attr)

    def message(self, x_j, edge_weight):
        # x_j E,D
        # edge_weight B,E,1
        return edge_weight * x_j

    def update(self, aggr_out):
        # aggr_out N,D
        return F.relu(self.lin_out(aggr_out))


