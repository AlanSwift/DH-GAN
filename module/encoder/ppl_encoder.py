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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VisualHintPredictor(nn.Module):
    def __init__(self, dim_ppl, dim_answer, hidden_size, dropout):
        super(VisualHintPredictor, self).__init__()
        self.ppl_in = nn.Linear(dim_ppl, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
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

        self.b = nn.ModuleList([Bottleneck(hidden_size, hidden_size, downsample=None, dilation=1 if _ <= 1 else 2) for _ in range(self.n_layer)])


    def forward(self, ppl, answer_vec):


        ppl_raw = self.ppl_in(ppl)
        ppl_raw = self.bn(ppl_raw.transpose(1, 2)).transpose(1, 2)

        ppl_f = ppl_raw.view(ppl_raw.shape[0], 6, 6, -1).permute(0, 3, 1, 2)
        ret = []
        x = ppl_f
        for i in range(self.n_layer):
            x = self.b[i](x)
            ret.append(x.permute(0, 2, 3, 1).view_as(ppl_raw))
        
        # out = self.b1(ppl_f)
        # print(out.shape)


        # A = self.topology_learner(ppl_raw)
        # _, ppl_enc = self.gnn(ppl_raw, A)

        # ppl_enc = self.obj_interaction(ppl_raw)
        p_list = [ppl_raw, *ret]

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



# class VisualHintPredictor(nn.Module):
#     def __init__(self, dim_ppl, dim_answer, hidden_size, dropout):
#         super(VisualHintPredictor, self).__init__()
#         self.ppl_in = nn.Linear(dim_ppl, hidden_size)
#         self.bn = nn.BatchNorm1d(hidden_size)
#         self.n_layer = 3
#         self.obj_interaction = Transformer(d_model=hidden_size, d_hidden=hidden_size, n_layers=self.n_layer, n_heads=6, drop_ratio=0.2)

#         self.attn = Attention(query_size=dim_answer, memory_size=hidden_size, attention_funtion="mlp",
#                               hidden_size=dim_answer, has_bias=True, dropout=dropout)

#         self.predictor = nn.Linear(hidden_size, 1)
#         self.ppl_with_ans = nn.Sequential(nn.Linear(hidden_size + dim_answer, hidden_size),
#                                             nn.ReLU(),
#                                             nn.Dropout(dropout))
#         self.ans2latent = nn.Sequential(nn.Linear(dim_answer, (self.n_layer+1)*hidden_size),
#                                         nn.Dropout(dropout))

#         self.topology_learner = GraphLeaner(k=10, d_in=hidden_size, d_hidden=hidden_size, epsilon=0.7)
#         self.gnn = GCNEncoder(d_in=hidden_size, d_out=hidden_size, dropout=dropout, layer=self.n_layer, step=36, need_align=False)

#         self.hybrid_prj = nn.Linear(hidden_size, hidden_size)
        

#     def forward(self, ppl, answer_vec):


#         ppl_raw = self.ppl_in(ppl)
#         ppl_raw = self.bn(ppl_raw.transpose(1, 2)).transpose(1, 2)


#         # A = self.topology_learner(ppl_raw)
#         # _, ppl_enc = self.gnn(ppl_raw, A)

#         ppl_enc = self.obj_interaction(ppl_raw)
#         p_list = [ppl_raw, *ppl_enc]

#         p_hybrid = torch.stack(p_list, dim=2) # [batch, num_ppl, cnt_head, dim]

#         filter_list = self.ans2latent(answer_vec).chunk(self.n_layer+1, dim=1)
#         filter = torch.stack(filter_list, dim=1) # [batch, cnt_head, dim]

#         filter = filter.unsqueeze(1)

#         logits = F.tanh(self.hybrid_prj(p_hybrid) + filter)
#         logits = self.predictor(logits).squeeze(3)
#         logits = logits.mean(2)
        
#         # logits = torch.mul(p_hybrid, filter.unsqueeze(1)).sum(-1).mean(-1)
#         return logits



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


class TransformerEncoder(nn.Module):
    def __init__(self, dim_ppl, dim_ans, hidden_size, dropout=0.2):
        super().__init__()
        self.dim_vh = 300
        self.vh_prj = nn.Linear(1, self.dim_vh)
        self.double_hints_ppl = nn.Sequential(nn.Linear(dim_ppl + self.dim_vh, hidden_size),
                                                nn.ReLU(),
                                                nn.Dropout(dropout))
        self.n_layer = 3
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        self.obj_interaction = transformer_encoder

        # self.obj_interaction = Transformer(d_model=hidden_size, d_hidden=hidden_size, n_layers=self.n_layer, n_heads=6, drop_ratio=0.2)
        

    def forward(self, ppl, visual_hint, answer):
        # vh = self.vh_prj(visual_hint.float().unsqueeze(2))
        # # print(ppl.shape, vh.shape)
        # ppl_feats = self.double_hints_ppl(torch.cat((ppl, vh), dim=2))
        # return ppl_feats
        # ppl_feats = self.double_hints_ppl(torch.cat((ppl, vh, answer.unsqueeze(1).expand(-1, vh.shape[1], -1)), dim=2))
        return self.obj_interaction(ppl)



class GNNEncoder(nn.Module):
    def __init__(self, dim_ppl, dim_ans, hidden_size, dropout=0.2):
        super().__init__()
        self.dim_vh = 300
        self.vh_prj = nn.Linear(1, self.dim_vh)
        self.double_hints_ppl = nn.Sequential(nn.Linear(dim_ppl + self.dim_vh, hidden_size),
                                                nn.ReLU(),
                                                nn.Dropout(dropout))
        self.n_layer = 3
        self.obj_interaction = GCNEncoder(d_in=dim_ppl, d_out=hidden_size, dropout=dropout, layer=self.n_layer, step=36, need_align=False)

        # self.obj_interaction = Transformer(d_model=hidden_size, d_hidden=hidden_size, n_layers=self.n_layer, n_heads=6, drop_ratio=0.2)
        

    def forward(self, ppl, visual_hint, answer):
        A = visual_hint.unsqueeze(2).expand(-1, -1, visual_hint.shape[1]).detach()
        enc, _ = self.obj_interaction(ppl, A)
        return enc

        # vh = self.vh_prj(visual_hint.float().unsqueeze(2))
        # # print(ppl.shape, vh.shape)
        # ppl_feats = self.double_hints_ppl(torch.cat((ppl, vh), dim=2))
        # return ppl_feats
        # ppl_feats = self.double_hints_ppl(torch.cat((ppl, vh, answer.unsqueeze(1).expand(-1, vh.shape[1], -1)), dim=2))
        return self.obj_interaction(ppl)


class DynamicGNNEncoder(nn.Module):
    def __init__(self, dim_ppl, dim_ans, hidden_size, dropout=0.2):
        super().__init__()
        self.dim_vh = 300
        self.vh_prj = nn.Linear(1, self.dim_vh)
        self.double_hints_ppl = nn.Sequential(nn.Linear(dim_ppl + self.dim_vh + dim_ans, hidden_size),
                                                nn.ReLU(),
                                                nn.Dropout(dropout))
        self.topology = GraphLeaner(k=10, d_in=hidden_size, epsilon=0.7)
        self.n_layer = 3
        self.obj_interaction = GCNEncoder(d_in=dim_ppl, d_out=hidden_size, dropout=dropout, layer=self.n_layer, step=36, need_align=False)

        # self.obj_interaction = Transformer(d_model=hidden_size, d_hidden=hidden_size, n_layers=self.n_layer, n_heads=6, drop_ratio=0.2)
        

    def forward(self, ppl, visual_hint, answer):
        vh_feats = self.vh_prj(visual_hint.detach())
        ppl_feats = self.double_hints_ppl(torch.cat((ppl, vh_feats, answer.unsqueeze(1).expand(-1, vh_feats.shape[1], -1)), dim=2))
        A = self.topology(ppl_feats)
        enc, _ = self.obj_interaction(ppl, A)
        return enc

        # vh = self.vh_prj(visual_hint.float().unsqueeze(2))
        # # print(ppl.shape, vh.shape)
        # ppl_feats = self.double_hints_ppl(torch.cat((ppl, vh), dim=2))
        # return ppl_feats
        # ppl_feats = self.double_hints_ppl(torch.cat((ppl, vh, answer.unsqueeze(1).expand(-1, vh.shape[1], -1)), dim=2))
        return self.obj_interaction(ppl)