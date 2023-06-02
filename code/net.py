import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherGNN(nn.Module):
    def __init__(self, args):
        super(WeatherGNN, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.factor_embed_net = nn.Linear(7,args.hidden_dim)
        self.fgnn = FactorGNN(self.hidden_dim,self.hidden_dim,2,args.embed_dim)
        # self.fusion = nn.Linear(args.factor_num*self.hidden_dim, args.embed_dim)
        self.m1 = FHMP(args.s1,args.s2,args.a1,args.a2,args.a3,args.neighbor_index_layer_2,args.neighbor_index_layer_3,args.factor_num*self.hidden_dim,self.num_layers)
        # self.m2 = FHMP(self.hidden_dim,self.num_layers)
        self.factor_embeddings = nn.Parameter(torch.randn(args.factor_num, args.embed_dim), requires_grad=True)
        self.decoder = nn.Linear(args.factor_num*self.hidden_dim,5)
        
        
    def forward(self,A_1_featurs):
        b,t,h,w,f = A_1_featurs.shape
        n = h*w
        A_1_featurs = A_1_featurs.reshape(b,t,h*w,f)
        x_ = A_1_featurs.permute(0,2,3,1)
        x_for_factorgnn = self.factor_embed_net(x_)
        x_factorgnn = self.fgnn(x_for_factorgnn,self.factor_embeddings) # B,N,F,E
        # x_fusion = self.fusion(x_factorgnn.reshape(b,n,-1))
        x_updated = self.m1((x_factorgnn.reshape(b,n,-1)))
        # x_updated_2 = self.m1(x_updated)
        x_updated = x_updated.reshape(b,n,-1)
        x_corr = self.decoder(x_updated)
        return x_corr


class FactorGNN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(FactorGNN, self).__init__()
        self.cheb_k = 2
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, factor_embeddings):
        #x shaped [B,N,F,E], factor_embeddings shaped [N,D] -> supports shaped [N, N]
        #output shape [B,N,F,E]
        node_num = factor_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(factor_embeddings, factor_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', factor_embeddings, self.weights_pool)
        bias = torch.matmul(factor_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bqmc->bqknc", supports, x) # B,N,K,F,E
        x_g = x_g.permute(0,1,3,2,4) # B,N,F,K,E
        x_gconv = torch.einsum('bqnki,nkio->bqno', x_g, weights)+bias # B,N,F,E
        return x_gconv
    


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



class MP(nn.Module):
    def __init__(self, s1,s2,a1,a2,a3,neighbor_index_layer_2,neighbor_index_layer_3,hidden_dim,num_layers):
        super(MP, self).__init__()
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

        self.s1 = s1
        self.s2 = s2
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.neighbor_index_layer_2 = neighbor_index_layer_2
        self.neighbor_index_layer_3 = neighbor_index_layer_3
    
    def forward(self, A_1_featurs):
        A_1_updated_features = self.sub_gcn(A_1_featurs,self.a1)
        return A_1_updated_features



class FHMP(nn.Module):
    def __init__(self, s1,s2,a1,a2,a3,neighbor_index_layer_2,neighbor_index_layer_3,hidden_dim,num_layers):
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

        self.s1 = s1
        self.s2 = s2
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.neighbor_index_layer_2 = neighbor_index_layer_2
        self.neighbor_index_layer_3 = neighbor_index_layer_3
        
        
    def forward(self, A_1_featurs):
        A_2_featurs = self.s1.T@A_1_featurs
        A_3_featurs = self.s2.T @ A_2_featurs
        A_features_sets = [A_1_featurs,A_2_featurs,A_3_featurs]

        # 局部计算动态矩阵
        # A_1_updated_features = 
        # for i in range(self.s1.shape[1]):
        #     # 得到每个子图的邻接矩阵
        #     sub_nodes_i = self.s1[:,i].nonzero().flatten()
        #     sub_graph_i = self.a1[sub_nodes_i][:, sub_nodes_i]
        #     sub_features_i = A_features_sets[0][:,sub_nodes_i,:]
        
        # A_dynamic_sets = [self.a1,self.a2,self.a3]

        A_1_dynamic = self.dynamic_adj(A_1_featurs,A_1_featurs)
        A_2_dynamic = self.s1.T@A_1_dynamic@self.s1
        A_3_dynamic = self.s2.T@A_2_dynamic@self.s2
        A_dynamic_sets = [A_1_dynamic,A_2_dynamic,A_3_dynamic]       

        # 对于每个子图
        A_1_updated_features = torch.zeros(A_1_featurs.shape).to(A_1_featurs.device)
        for i in range(self.s1.shape[1]):
            # 得到每个子图的邻接矩阵
            sub_nodes_i = self.s1[:,i].nonzero().flatten()
            sub_graph_i = self.a1[sub_nodes_i][:, sub_nodes_i]
            sub_features_i = A_features_sets[0][:,sub_nodes_i,:]
            # 接受来自最底层子图邻居的信息 l=1
            messages_from_layer_1_i = self.sub_gcn(sub_features_i,sub_graph_i)
            
            # 接受来自其他层非祖先节点的消息 l=2,3
            # 第二层祖先节点的索引，其实就是i
            # 第二层祖先节点的邻居
            neighbor_index_layer_2_i = self.neighbor_index_layer_2[i]
            # 根据第二层动态边 聚合第二层祖先节点的邻居信息
            neighbor_info_layer_2_i = A_features_sets[1][:,neighbor_index_layer_2_i,:]
            messages_from_layer_2_i = A_dynamic_sets[1][i,neighbor_index_layer_2_i].unsqueeze(0) @ neighbor_info_layer_2_i # torch.Size([64, 1, 64])
            
            # 第三层祖先节点的索引
            ancestor_index_layer_3_i = self.s2[i,:].nonzero().flatten().item()
            # 根据第三层动态边 聚合第三层祖先节点的邻居信息
            neighbor_index_layer_3_i = self.neighbor_index_layer_3[ancestor_index_layer_3_i]
            neighbor_info_layer_3_i = A_features_sets[2][:,neighbor_index_layer_3_i,:]
            messages_from_layer_3_i = A_dynamic_sets[2][ancestor_index_layer_3_i,neighbor_index_layer_3_i].unsqueeze(0) @ neighbor_info_layer_3_i

            # 聚合不同层的邻居信息 并更新
            aggregated_message_i = self.aggregator(messages_from_layer_1_i, messages_from_layer_2_i, messages_from_layer_3_i)
            updated_info_i = self.updator(sub_features_i, aggregated_message_i)
            A_1_updated_features[:,sub_nodes_i,:] = updated_info_i
        
        return A_1_updated_features

