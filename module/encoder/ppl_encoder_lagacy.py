import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from module.submodule.graph_learner import GraphLeaner
from module.submodule.gcn import GCN, GCNEncoder
from module.submodule import str2gnn
import math
import numpy as np
from module.encoder.transformer import Transformer
from module.submodule.attention import Attention


class VisualHintPredictor(nn.Module):
    def __init__(self, dim_ppl, dim_answer, hidden_size, dropout):
        super(VisualHintPredictor, self).__init__()
        self.ppl_in = nn.Linear(dim_ppl, hidden_size)
        self.bn = nn.SyncBatchNorm(hidden_size)
        self.n_layer = 3
        self.obj_interaction = Transformer(d_model=hidden_size, d_hidden=hidden_size, n_layers=self.n_layer, n_heads=6, drop_ratio=0.2)

        self.attn = Attention(query_size=dim_answer, memory_size=hidden_size, attention_funtion="mlp",
                              hidden_size=dim_answer, has_bias=True, dropout=dropout)

        self.predictor = nn.Linear(hidden_size, 1)
        self.ppl_with_ans = nn.Sequential(nn.Linear(hidden_size + dim_answer, hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(dropout))
        self.ans2latent = nn.Sequential(nn.Linear(dim_answer, (self.n_layer+1)*hidden_size),
                                        nn.Dropout(dropout))

        self.topology_learner = GraphLeaner(k=10, d_in=hidden_size, d_hidden=hidden_size, epsilon=0.7)
        self.gnn = GCNEncoder(d_in=hidden_size, d_out=hidden_size, dropout=dropout, layer=self.n_layer, step=36, need_align=False)

        self.hybrid_prj = nn.Linear(hidden_size, hidden_size)
        

    def forward(self, ppl, answer_vec):


        ppl_raw = self.ppl_in(ppl)
        ppl_raw = self.bn(ppl_raw.transpose(1, 2)).transpose(1, 2)


        # A = self.topology_learner(ppl_raw)
        # _, ppl_enc = self.gnn(ppl_raw, A)

        ppl_enc = self.obj_interaction(ppl_raw)
        p_list = [ppl_raw, *ppl_enc]

        p_hybrid = torch.stack(p_list, dim=2) # [batch, num_ppl, cnt_head, dim]

        filter_list = self.ans2latent(answer_vec).chunk(self.n_layer+1, dim=1)
        filter = torch.stack(filter_list, dim=1) # [batch, cnt_head, dim]

        filter = filter.unsqueeze(1)

        logits = F.tanh(self.hybrid_prj(p_hybrid) + filter)
        logits = self.predictor(logits).squeeze(3)
        logits = logits.mean(2)
        
        # logits = torch.mul(p_hybrid, filter.unsqueeze(1)).sum(-1).mean(-1)
        return logits



        # answer_feats = answer_hint

        # ppl_aligned, agg = self.align(ppl, answer_feats, answer_mask)
        # ppl = self.ppl_with_ans(torch.cat((ppl, answer_vec.unsqueeze(1).expand(-1, ppl.shape[1], -1)), dim=2))

        # ppl_encoded = self.obj_interaction(ppl)

        # attned, _ = self.attn(query=answer_vec, memory=ppl_encoded)
        # vh_logits = self.predictor(ppl)
        return vh_logits
        exit(0)

        # latent = self.visual_hint_classifier_fc[0](student_latent)
        # visual_hint_prob_student = self.visual_hint_classifier_fc[1](latent)
        # pos_pred = self.latent_to_pos(latent)
        # ans_pred = self.graph_pool(latent)

        # vh = latent

        # # ppl_enhanced, _ = self.align_all(ppl, loc_feats, answer_feats, vh, answer_mask)  # ppl, loc, answer, vh,
        # ppl_enhanced = torch.cat((ppl, agg, vh), dim=-1)
        # ppl_enhanced = self.mlp(ppl_enhanced)
        # adj = self.graph_learner(vh)

        # ppl_student = self.gnn_student(ppl_enhanced, adj, answer_hint)

        return ppl_student, visual_hint_prob_student, pos_pred, ans_pred, adj
