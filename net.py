import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMask(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AttentionMask, self).__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k):
        q = self.q_linear(q)
        k = self.k_linear(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        attention_mask = F.softmax(scores, dim=-1)
        attention_mask = self.dropout(attention_mask)        
        # 输出 NxN
        return torch.mean(attention_mask,dim=0) 


class SubGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SubGCN, self).__init__()
        self.mlp = nn.Linear(in_features, hidden_features)
    def forward(self,x,A):
        conv = self.mlp(torch.einsum('nk,bnd->bkd',(A,x)))
        out = F.relu(conv)
        return out


class AggMessage(nn.Module):
    def __init__(self, dim, num_layers):
        super(AggMessage, self).__init__()
        self.mlp = nn.Linear(dim*num_layers, 1)
    def forward(self,m1,m2,m3):
        m2 = m2.repeat(1,m1.shape[1],1)
        m3 = m3.repeat(1,m1.shape[1],1)
        out = self.mlp(torch.cat([m1,m2,m2], dim=-1))
        out = F.relu(out)
        return out


class UpInfo(nn.Module):
    def __init__(self, dim):
        super(UpInfo, self).__init__()
        self.mlp = nn.Linear(dim, dim)
    def forward(self,self_info,aggregated_message):
        out = F.relu(self.mlp(self_info+aggregated_message))
        return out


class FHMP(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(FHMP, self).__init__()
        # self.hidden_dim = hidden_dim
        
        # 动态的邻接矩阵
        self.dynamic_adj = AttentionMask(hidden_dim)
        # 最底层 子图内部做message passing
        self.sub_gcn = SubGCN(hidden_dim, hidden_dim, hidden_dim)
        # sub_gcn_sets = nn.ModuleList()
        # for _ in range(0, num_sub_graph):
        #     sub_gcn_sets.append(SubGCN(hidden_dim, hidden_dim, hidden_dim))
        self.aggregator = AggMessage(hidden_dim, num_layers)
        self.updator = UpInfo(hidden_dim)
        
        
    def forward(self, A_1_featurs, S_set, A_sets,neighbor_index_layer_2, neighbor_index_layer_3):
        A_2_featurs = S_set[0].T@A_1_featurs
        A_3_featurs = S_set[1].T @ A_2_featurs
        A_features_sets = [A_1_featurs,A_2_featurs,A_3_featurs]


        A_1_dynamic = self.dynamic_adj(A_1_featurs,A_1_featurs)
        A_2_dynamic = S_set[0].T@A_1_dynamic@S_set[0]
        A_3_dynamic = S_set[1].T@A_2_dynamic@S_set[1]
        A_dynamic_sets = [A_1_dynamic,A_2_dynamic,A_3_dynamic]       

        # 对于每个子图
        A_1_updated_features = torch.zeros(A_1_featurs.shape).to(A_1_featurs.device)
        for i in range(S_set[0].shape[1]):
            # 得到每个子图的邻接矩阵
            sub_nodes_i = S_set[0][:,i].nonzero().flatten()
            sub_graph_i = A_sets[0][sub_nodes_i][:, sub_nodes_i]
            sub_features_i = A_features_sets[0][:,sub_nodes_i,:]
            # 接受来自最底层子图邻居的信息 l=1
            messages_from_layer_1_i = self.sub_gcn(sub_features_i,sub_graph_i)
            
            # 接受来自其他层非祖先节点的消息 l=2,3
            # 第二层祖先节点的索引，其实就是i
            # 第二层祖先节点的邻居
            neighbor_index_layer_2_i = neighbor_index_layer_2[i]
            # 根据第二层动态边 聚合第二层祖先节点的邻居信息
            neighbor_info_layer_2_i = A_features_sets[1][:,neighbor_index_layer_2_i,:]
            messages_from_layer_2_i = A_dynamic_sets[1][i,neighbor_index_layer_2_i].unsqueeze(0) @ neighbor_info_layer_2_i # torch.Size([64, 1, 64])
            
            # 第三层祖先节点的索引
            ancestor_index_layer_3_i = S_set[1][i,:].nonzero().flatten().item()
            # 根据第三层动态边 聚合第三层祖先节点的邻居信息
            neighbor_index_layer_3_i = neighbor_index_layer_3[ancestor_index_layer_3_i]
            neighbor_info_layer_3_i = A_features_sets[2][:,neighbor_index_layer_3_i,:]
            messages_from_layer_3_i = A_dynamic_sets[2][ancestor_index_layer_3_i,neighbor_index_layer_3_i].unsqueeze(0) @ neighbor_info_layer_3_i

            # 聚合不同层的邻居信息 并更新
            aggregated_message_i = self.aggregator(messages_from_layer_1_i, messages_from_layer_2_i, messages_from_layer_3_i)
            updated_info_i = self.updator(sub_features_i, aggregated_message_i)
            A_1_updated_features[:,sub_nodes_i,:] = updated_info_i
        
        return A_1_updated_features

