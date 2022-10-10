import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphLeaner(nn.Module):
    def __init__(self, k, d_in, d_hidden, epsilon=0.75):
        super(GraphLeaner, self).__init__()
        self.k = k
        self.l_in = nn.Linear(d_in, d_hidden)
        self.l_out = nn.Linear(d_in, d_hidden)

        self.head = 3
        self.epsilon = epsilon

        self.weight_tensor_1 = torch.Tensor(self.head, d_in)
        self.weight_tensor_2 = torch.Tensor(self.head, d_in)
        self.weight_tensor_1 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_1))
        self.weight_tensor_2 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_2))


        self.linear_sims1 = nn.ModuleList([nn.Linear(d_in, 1, bias=False) for _ in range(self.head)])
        self.linear_sims2 = nn.ModuleList([nn.Linear(d_in, 1, bias=False) for _ in range(self.head)])
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, feats: torch.Tensor):
        """
            Learn the graph topology
        Parameters
        ----------
        feats: torch.Tensor
            shape=[B, N, D]
        Returns
        -------
        adj: torch.Tensor
            shape=[B, N, N]
        """
        # sim = self.static_cos_score(feats)
        # sim = self.attention(feats)
        sim = self.weighted_cos_score(feats)
        # sim = self.gat(feats)

        # ret = self.get_knn_adj(sim, self.k)
        ret = self.get_epsilon_adj(sim, self.epsilon)
        return ret

    def get_knn_adj(self, scores, k):
        # _, idx = torch.sort(sim, descending=True)
        # _[:, :, self.k:] = 0
        # ret = torch.zeros(feats.shape[0], feats.shape[1], feats.shape[1]).cuda()
        # ret.scatter_(dim=2, index=idx, src=_)
        knn_val, knn_ind = torch.topk(scores, k, dim=-1)
        ret = (0 * torch.ones_like(scores)).cuda().scatter_(-1, knn_ind, knn_val)
        return ret

    def get_epsilon_adj(self, scores, epsilon=0.75):
        # mask = (scores > epsilon).detach().float()
        # scores = scores * mask + 0 * (1 - mask)
        scores[scores < epsilon] = 0
        return scores

    def static_cos_score(self, feats):
        xx = torch.matmul(feats, feats.transpose(-2, -1))
        norm = feats.norm(p=2, dim=-1, keepdim=True)
        sim = xx / (norm * norm.transpose(-2, -1)).clamp(min=1e-8)
        return sim

    def weighted_cos_score(self, feats):
        expand_weight_tensor_1 = self.weight_tensor_1.unsqueeze(1)
        expand_weight_tensor_2 = self.weight_tensor_2.unsqueeze(1)

        if len(feats.shape) == 3:
            expand_weight_tensor_1 = expand_weight_tensor_1.unsqueeze(1)
            expand_weight_tensor_2 = expand_weight_tensor_2.unsqueeze(1)

        context_fc_1 = feats.unsqueeze(0) * expand_weight_tensor_1
        context_fc_2 = feats.unsqueeze(0) * expand_weight_tensor_2
        context_norm_1 = F.normalize(context_fc_1, p=2, dim=-1)
        context_norm_2 = F.normalize(context_fc_2, p=2, dim=-1)

        attention = torch.matmul(context_norm_1, context_norm_2.transpose(-1, -2)).mean(0)
        return attention

    def attention(self, feats):
        feats_in = F.relu(self.l_in(feats))
        feats_out = F.relu(self.l_out(feats))
        attention = torch.matmul(feats_in, feats_out.transpose(-1, -2)) / math.sqrt(feats_in.shape[-1])
        # attention = F.softmax(attention, dim=-1)
        line_max, _ = torch.max(attention, dim=-1)
        line_max = line_max.detach()
        line_max[line_max == 0] = 1.
        attention = attention / (line_max.unsqueeze(2))
        return attention

    def gat(self, feats):
        attention = []
        for _ in range(len(self.linear_sims1)):
            a_input1 = self.linear_sims1[_](feats)
            a_input2 = self.linear_sims2[_](feats)
            attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

        attention = torch.mean(torch.stack(attention, 0), 0)
        return attention