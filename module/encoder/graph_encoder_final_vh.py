import torch
import torch.nn as nn
from module.submodule.graph_learner import GraphLeaner
from module.submodule import str2gnn
import math
import numpy as np

class DeepCrossModalAlign(nn.Module):
    def __init__(self, dim_ppl, dim_answer, hidden_size, dim_out, dropout=0.2):
        super(DeepCrossModalAlign, self).__init__()
        self.align_ppl = nn.Linear(dim_ppl, hidden_size)
        self.align_answer = nn.Linear(dim_answer, hidden_size)
        self.w = nn.Linear(hidden_size, 1)
        self.out_proj = nn.Linear(dim_ppl + dim_answer, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ppl, loc, answer, answer_mask=None):

        aligned_stud = self.align(ppl=ppl, answer=answer, answer_mask=answer_mask)
        return aligned_stud

    def align(self, ppl, answer, answer_mask=None):
        ppl_align = self.align_ppl(ppl)
        ans = self.align_answer(answer)
        scores = ppl_align.unsqueeze(2) + ans.unsqueeze(1)
        scores = self.w(torch.tanh(scores))
        if answer_mask is not None:
            answer_length = answer.shape[1]
            answer_mask = answer_mask[:, :answer_length]
            answer_mask = answer_mask.unsqueeze(1).unsqueeze(-1).expand(-1, ppl.shape[1], -1, -1)
            scores = scores.masked_fill((1 - answer_mask.float()).bool(), -float("inf"))


        scores = torch.softmax(scores.squeeze(3), dim=-1)
        agg = torch.bmm(scores, answer)

        aligned_feats = torch.cat((ppl, agg), dim=-1)
        aligned_feats = self.out_proj(aligned_feats)
        aligned_feats = torch.relu(aligned_feats)
        aligned_feats = self.dropout(aligned_feats)
        return aligned_feats, agg



class GraphPool(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(GraphPool, self).__init__()
        self.cnn_out_dim = 10*1*60
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=[1, 10],
                                           dilation=[1, 3], stride=[1, 2]),
                                 nn.MaxPool2d(kernel_size=[100, 4]),
                                 nn.ReLU(),
                                 nn.Dropout(0.3))
        self.cls = nn.Linear(self.cnn_out_dim, d_out)


    def forward(self, g):
        g = g.unsqueeze(1)
        g = self.cnn(g)
        g = g.view(g.shape[0], -1)
        ret = self.cls(g)
        return ret



class Attention(nn.Module):
    def __init__(self, query_size, memory_size, hidden_size, has_bias=False, attention_funtion="mlp", dropout=0.2):
        super(Attention, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.attn_type = attention_funtion
        assert self.attn_type in ["mlp", "general"]
        if self.attn_type == "general":
            self.query2memory = nn.Linear(self.query_size, self.memory_size)
        elif self.attn_type == "mlp":
            self.query_in = nn.Linear(self.query_size, self.hidden_size, bias=True if has_bias else False)
            self.memory_in = nn.Linear(self.memory_size, self.hidden_size, bias=False)
            self.out = nn.Linear(self.hidden_size, 1, bias=False)
        else:
            raise NotImplementedError()

        self.dropout = nn.Dropout(dropout)
        self.inf = 1e8




    def forward(self, query, memory, memory_mask=None, coverage=None):
        """
            Attention function
        Parameters
        ----------
        query: torch.Tensor, shape=[B, M, D]
        memory: torch.Tensor, shape=[B, N, D]
        memory_mask: torch.Tensor, shape=[B, M, N]
        coverage: torch.Tensor, shape=[B, N, D]

        Returns
        -------
        attn_scores: torch.Tensor, shape=[B, M, N]
        """
        assert len(query.shape) == 3
        assert len(memory.shape) == 3
        assert query.shape[0] == memory.shape[0]
        if coverage is not None:
            assert len(coverage.shape) == 3
            assert coverage.shape[-1] == self.hidden_size
            assert coverage.shape == memory.shape

        aligns = self._calculate_aligns(query, memory, coverage=coverage)
        scores = torch.softmax(aligns, dim=-1)
        ret = torch.matmul(self.dropout(scores), memory)


        return ret

    def _calculate_aligns(self, src, tgt, coverage=None):
        """
            Attention score calculation.
        Parameters
        ----------
        src: torch.Tensor, shape=[B, M, D]
        tgt: torch.Tensor, shape=[B, N, D]
        coverage: torch.Tensor, shape=[B, N, D]
        Returns
        -------
        aligns: torch.Tensor, shape=[B, M, N]
        """
        if self.attn_type == "mlp":
            src = self.query_in(src)
            tgt = self.memory_in(tgt)
            aligns = src.unsqueeze(2) + tgt.unsqueeze(1)
            aligns = torch.tanh(aligns)
            aligns = self.out(aligns)
            aligns = aligns.view(aligns.shape[0], aligns.shape[1], aligns.shape[2])
        elif self.attn_type == "general":
            src = self.query2memory(src)
            src = torch.tanh(src)
            aligns = torch.bmm(src, tgt.transpose(1, 2))
            aligns = aligns.contiguous()
        else:
            raise NotImplementedError()
        return aligns


class GraphEncoderVisualHint(nn.Module):
    def __init__(self, gnn, dim_ppl, dim_loc_feats, answer_amount, dim_visual_hint, dim_answer, hidden_size, topk, dropout, ppl_num,
                 epsilon=0.75):
        super(GraphEncoderVisualHint, self).__init__()
        self.loc_fc = nn.Sequential(nn.Linear(4, 200),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))

        self.loc_fc_small = nn.Sequential(nn.Linear(4, 200),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))

        self.latent_dim = hidden_size

        self.visual_hint_classifier_latent = nn.Sequential(nn.Linear(dim_ppl + dim_loc_feats + dim_answer, self.latent_dim),
                                                           nn.ReLU(),
                                                           nn.Dropout(dropout))

        self.answer_number = answer_amount  # 16367

        self.dim_vh = 512

        self.ans_cls = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(hidden_size, self.answer_number))

        self.latent_to_pos = nn.Sequential(nn.Linear(self.dim_vh, 4), nn.ReLU())


        self.visual_hint_classifier_fc = nn.ModuleList([nn.Sequential(nn.Linear(self.latent_dim, self.dim_vh),
                                                                      nn.ReLU(),
                                                                      nn.Dropout()),
                                                        nn.Linear(self.dim_vh, 2)])

        self.graph_learner = GraphLeaner(k=topk, d_in=self.dim_vh, d_hidden=self.dim_vh, epsilon=epsilon)
        self.gnn_student = str2gnn[gnn](d_in=hidden_size, d_out=hidden_size, step=ppl_num, layer=2, dropout=dropout,
                                d_answer=dim_answer, need_align=False)

        self.graph_pool = GraphPool(d_in=self.dim_vh, d_hidden=hidden_size, d_out=self.answer_number)


        self.align = DeepCrossModalAlign(dim_ppl=dim_ppl + 200 + 200,  dim_answer=dim_answer,
                                         hidden_size=dim_answer, dim_out=hidden_size, dropout=dropout)


        self.vh_prj = nn.Sequential(nn.Linear(1, self.dim_vh),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))

        self.dim_ppl_enhanced = dim_ppl + self.dim_vh + dim_answer + 200


        self.cls_enc = nn.Sequential(nn.Linear(1, 200),
                                     nn.ReLU(),
                                     nn.Dropout())

        self.attn = Attention(query_size=dim_ppl, memory_size=hidden_size, attention_funtion="mlp",
                              hidden_size=128, has_bias=True, dropout=dropout)

        self.mlp = nn.Sequential(nn.Linear(dim_ppl + 200 + 200 + dim_answer + self.dim_vh, hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(dropout))



    def forward(self, ppl, ppl_infos, answer_hint, answer_mask, visual_hint=None, img_feats=None, teacher_forcing=0, training=True):
        loc_input = ppl_infos[:, :, 0:4]
        loc_feats = self.loc_fc(loc_input)
        origin_ppl = ppl

        cls_feats = self.cls_enc(ppl_infos[:, :, 4].unsqueeze(2).float())
        ppl = torch.cat((ppl, cls_feats, loc_feats), dim=-1)

        answer_feats = answer_hint

        student_latent, agg = self.align(ppl, loc_feats, answer_feats, answer_mask)

        latent = self.visual_hint_classifier_fc[0](student_latent)
        visual_hint_prob_student = self.visual_hint_classifier_fc[1](latent)
        pos_pred = self.latent_to_pos(latent)
        ans_pred = self.graph_pool(latent)

        vh = latent

        # ppl_enhanced, _ = self.align_all(ppl, loc_feats, answer_feats, vh, answer_mask)  # ppl, loc, answer, vh,
        ppl_enhanced = torch.cat((ppl, agg, vh), dim=-1)
        ppl_enhanced = self.mlp(ppl_enhanced)
        adj = self.graph_learner(vh)

        ppl_student = self.gnn_student(ppl_enhanced, adj, answer_hint)

        return ppl_student, visual_hint_prob_student, pos_pred, ans_pred, adj
