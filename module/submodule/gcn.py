import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
"""
1, 1, 0
0, 0, 0
1, 0, 1
"""


class CrossGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc_gate1 = nn.Linear(d_model, d_model, bias=False)
        self.fc_gate2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x1, x2):
        g1 = torch.sigmoid(self.fc_gate1(x1))
        x2_ = g1 * x2
        g2 = torch.sigmoid(self.fc_gate2(x2))
        x1_ = g2 * x1
        return x1_, x2_

class GCN(nn.Module):
    def __init__(self, d_in, d_out):
        super(GCN, self).__init__()
        self.fc = nn.Linear(d_in, d_out, bias=True)

    def forward(self, x, adj):
        n_node = x.shape[1]
        b = x.shape[0]
        A=adj
        # A = (adj + adj.transpose(-1, -2)) / 2.0
        A = A + torch.eye(n_node).cuda().float()
        A = A.float()
        #A = ((adj + adj.transpose(-1, -2)) > 0).float() + torch.eye(n_node).cuda().float()
        D = torch.sum(A, -1) # [b, 1000]
        mask = D == 0

        D = 1. / torch.sqrt(D)
        D.masked_fill_(mask=mask, value=0)

        D_inv = torch.diag_embed(D)

        # D = D.view(b, n_node, 1).expand(-1, -1, n_node)
        # D = 1. / torch.sqrt(D)
        #
        # D_inv = torch.eye(n_node).unsqueeze(0).expand(b, n_node, n_node).cuda() * D

        w = torch.matmul(D_inv, A)
        w = torch.matmul(w, D_inv)
        x = torch.matmul(w, x)
        x = self.fc(x)
        x = nn.ReLU()(x)

        return x


class GCNEncoder(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.5, layer=2, step=50, need_align=True):
        super(GCNEncoder, self).__init__()
        self.gcn_layer = layer
        layers = []
        norm_layers = []
        for i in range(self.gcn_layer):
            layers.append(GCN(d_in=d_in, d_out=d_in))
            norm_layers.append(nn.LayerNorm([step, d_in]))
        self.layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.dropout = nn.Dropout(dropout)
        self.need_align = need_align
        if self.need_align:
            self.out_linear = nn.Linear(d_in, d_out)
            self.activate = nn.ReLU()

    def forward(self, graph, adj):

        x = graph
        ret = []
        for layer, norm in zip(self.layers, self.norm_layers):
            res = x
            x = layer(x, adj)
            x = (res + x) / math.sqrt(2)
            x = norm(x)
            x = self.dropout(x)
            ret.append(x)
        if self.need_align:
            x = self.out_linear(x)
            x = self.activate(x)
        return x, ret
