import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch.nn.functional as F


class EGCN(torch.nn.Module):
    def __init__(self, activation, device='cuda', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})
        emb_dim = 64
        feats = [emb_dim,  # 定义特征数量（2层）
                 emb_dim * 2,
                 emb_dim]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1, len(feats)):
            GRCU_args = u.Namespace({'in_feats': feats[i - 1],  # 输入特征
                                     'out_feats': feats[i],  # 输出特征
                                     'activation': activation})  # 激活函数

            grcu_i = GRCU(GRCU_args)  # 整个EvolveGCN
            # print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self, A_list, Nodes_list):
        node_feats = Nodes_list[-1]  # 获取最后一层特征

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)  # ,nodes_mask_list) 邻接矩阵；度节点矩阵（特征矩阵）

        out = Nodes_list  # [-1]最后一个时间步为最终输出节点
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)  # use node_feats.to_dense() if 2hot encoded input
        return out


class GRCU(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)  # 训练权重矩阵的RNN函数

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats, self.args.out_feats))  # [162,100];GCN初始权重
        self.reset_param(self.GCN_init_weights)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list):  # ,mask_list):
        GCN_weights = self.GCN_init_weights  # 初始化的GCN_weight
        out_seq = []
        for t, Ahat in enumerate(A_list):  # 遍历每个时间步邻接矩阵
            node_embs = node_embs_list[t]  # 获取该时间步的节点特征
            # first evolve the weights from the initial and use the new weights with the node_embs

            # RNN确定下一时间步的GCN_weight
            GCN_weights = self.evolve_weights(GCN_weights)  # ,node_embs,mask_list[t])
            # GCN
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))  # (N,dim)

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,  # 定义GRU中的Zt
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,  # 定义GRU中的Rt
                                  args.cols,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,  # Ht_hat
                                   args.cols,
                                   torch.nn.Tanh())

    def forward(self, prev_Q):  # ,prev_Z,mask):

        z_topk = prev_Q  # prev_Q:上一层的W(GCN_weight); Xt

        update = self.update(z_topk, prev_Q)  # Xt,Ht-1 => Zt
        reset = self.reset(z_topk, prev_Q)  # Xt,Ht-1 => Rt

        h_cap = reset * prev_Q  # Rt * Ht-1
        h_cap = self.htilda(z_topk, h_cap)  # Ht_hat

        new_Q = (1 - update) * prev_Q + update * h_cap  # (1-Zt) * Ht-1 + Zt * Ht_hat

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()
