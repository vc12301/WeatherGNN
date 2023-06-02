import torch
import torch.nn as nn
import torch.nn.functional as F

class FHGNN(nn.Module):
    def __init__(self, num_levels, dims):
        super().__init__()
        self.num_levels = num_levels
        self.dims = dims
        self.w
        self.w_q
        self.w_k
        self.gnn_layers = nn.ModuleList([FHGNNLayer(dim=self.dim) for i in range(self.num_levels)])
        
        
    def forward(self, x, assignment_list, reversed_assignment_list, edge_index_list, edge_weight_list):
    # x: [B, N, d]
    # assignment_list: the list of assginments of each level, where the l-th assignment is an N^(l) dimensional vector (the sparse format of assginment matrix) (L-1)
    # reversed_assignment_list: the list of reversed assginments of each level (l>0) to level 0, where each reversed assginment is an N-d vector (L-1)
    # edge_index_list: the list of edge index of each level, where the l-the edge index: [E, 2], corresponding to the graph of l-th level (L)
    # edge_weight_list: the corresponding predefined edge weights, where the l-the edge weight: [E, 1] (L)
    
        h, h_q, h_k = x * w, x * w_q, x * w_k
        h_prime = x + h
        h_prime += self.gnn_layers[0](h, h_q, h_k, edge_index_list[0], edge_weight_list[0])
        for i in range(self.num_levels - 1):
            h = self.get_high(h, assginment_list[i])
            # h * w might be here 
            h_q = self.get_high(h_q, assginment_list[i])
            h_k = self.get_high(h_k, assginment_list[i])
            h_hat = self.gnn_layers[0](h, h_q, h_k, edge_index_list[i+1], edge_weight_list[i+1])
            h_prime += self.duplication(h_hat, reversed_assignment_list[i])
            
        return h_prime

    def get_high(self, h, s)
        # should not be implemented with matrix multiplication, but with non-parametric message passing
        return h_high
    
    def duplication(self, h, s_reversed):
        # should not be implemented with matrix multiplication, but with indexing operation
        # h_0 = h[s_reversed]
        return h_0
        

    

# 每一层的消息传递
class FHGNNLayer(MessagePassing):
    def __init__(self, dim):
        super().__init__(aggr="add")
        self.dim = dim
        

    def forward(self, x, h_q, h_k edge_index, edge_attr):
        # x, h_q, h_k: [B, N, d]
        # edge_index: [B, E, 2]
        # edge_attr: [B, E, 1]
        # dynamic adjustment
        edge_index_q = edge_index[...,0].reshape(B,E)
        edge_index_k = edge_index[...,1].reshape(B,E)
        h_q_gather = h_q.gather(dim=1, index=edge_index_q) # [B, E, d]
        h_k_gather = h_k.gather(dim=1, index=edge_index_k) # [B, E, d]
        sim = torch.einsum('bed,bed->be', h_q_gather, h_k_gather).unsqueeze(2) / torch.sqrt(d)
        edge_attr *= sim
        edge_attr = edge_attr.exp()
        #计算softmax归一化因子
        norm_factor = torch.zeros((x.shape[0], x.shape[1], 1)) # [B, N ,1]
        norm_factor = norm_factor.index_add_(dim=1, index=edge_index_q, source=edge_attr) #(B,N,1)
        norm_factor_gather = norm_factor.gather(dim=1, index=edge_index_q) #(B,E,1)
        edge_attr /= norm_factor_gather
        
        return self.propagate(edge_index = edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr * x_j
    
    def aggregate(self):
        return
        
    def update(self):
        # nonlinear