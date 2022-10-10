from os.path import join
from module import encoder
import torch
import torch.nn as nn
from module.encoder.transformer import Transformer
from module.encoder.text_encoder_discriminator import EncoderRNNDifferentiable
from module.encoder.text_encoder import EncoderRNN
from module.submodule.attention import Attention
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Aggregation(nn.Module):
    def __init__(self, d_query, d_memory, d_model, dropout=0.2):
        super().__init__()
        self.query_in = nn.Linear(d_query, d_model)
        self.memory_in = nn.Linear(d_memory, d_model)
        self.wst = nn.Linear(d_model, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, memory_mask=None):
        if memory_mask is not None:
            lens = memory_mask.float().sum(1)
            max_lens = lens.max()
            memory = memory[:, :max_lens.long().item()]
            memory_mask = memory_mask[:, :max_lens.long().item()]
        query_enc = self.query_in(query)
        memory_enc = self.memory_in(memory)

        item = query_enc.unsqueeze(2) + memory_enc.unsqueeze(1)  # [nb, len1, len2, d]
        S = self.wst(torch.tanh(item)).squeeze(-1)  # [nb, len1, len2]
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
            S = S.masked_fill(~memory_mask.bool(), -float("inf"))
        S = self.dropout(torch.softmax(S, dim=-1))
        return torch.matmul(S, memory)  # [nb, len1, d]

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class Discriminator(nn.Module):
    def __init__(self, vocab, d_v, d_word, d_model, num_ans_candidates, dropout=0.2):
        super().__init__()
        n_layers = 2
        n_heads = 6
        attn_drop = dropout
        self.d_v = d_v
        self.d_model = d_model
        self.v_in = nn.Sequential(nn.Linear(self.d_v, self.d_model),
                                nn.LeakyReLU(0.1),
                                nn.Dropout(attn_drop))
        self.text_encoder = EncoderRNN(vocab=vocab, input_size=d_word, hidden_size=d_word, rnn_type="lstm", bidirectional=True, rnn_dropout=dropout)
        self.ans_encoder = EncoderRNN(vocab=vocab, input_size=d_word, hidden_size=d_word, rnn_type="gru", bidirectional=False, rnn_dropout=dropout)

        
        self.spatial_feats = nn.Sequential(nn.Linear(6, 300),
                                            nn.LeakyReLU(0.1),
                                            nn.Dropout(dropout))
        self.vh_feats = nn.Sequential(nn.Linear(1, 300),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(dropout))
        self.v_feats = nn.Sequential(nn.Linear(d_v + 300, d_model),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(dropout))
        self.interaction_v_q = Aggregation(d_query=d_model, d_memory=d_word, d_model=d_word)
        self.interaction_v_a = Aggregation(d_query=d_model, d_memory=d_word, d_model=d_word)
        self.vis_emb = nn.Sequential(nn.Linear(self.d_model + 1 * d_word, self.d_model),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout())
        self.pred = nn.Sequential(nn.Linear(self.d_model + d_word*2, self.d_model),
                                nn.Tanh(),
                                nn.Dropout(dropout),
                                nn.Linear(self.d_model, 1))
        self.loss_func = nn.CrossEntropyLoss()
        self.attn = Attention(query_size=d_word*2, memory_size=d_model, hidden_size=d_model, has_bias=True)

        self.neg_attn = Attention(query_size=d_word*3, memory_size=d_model, hidden_size=d_model, has_bias=True)
    
        self.q_net = FCNet([2*d_word, d_model])
        self.v_net = FCNet([d_model, d_model])

        self.classifier = SimpleClassifier(
        d_model, d_model * 2, num_ans_candidates, 0.5)

    def forward(self, visual_feats, visual_spatial_feats, visual_hints, question, question_mask, answer, labels=None):
        spatial_feats = self.spatial_feats(visual_spatial_feats)
        v_feats = torch.cat((visual_feats, spatial_feats), dim=2)
        v_feats = self.v_feats(v_feats)
        q_lens = question_mask.sum(1)
        q_lens[q_lens == 0] = 1
        q_feats, q_vec, _ = self.text_encoder(question)
        q_vec = torch.cat((q_vec[0].squeeze(0), q_vec[1].squeeze(0)), dim=1)

        attn_res, _ = self.attn(query=q_vec, memory=v_feats, memory_mask=None)

        q = self.q_net(q_vec)
        v = self.v_net(attn_res)
        joint_repr = q * v
        logits = self.classifier(joint_repr)
        if answer is not None:
            loss = self.loss_func(logits, answer.squeeze(1).long())
            return loss
        else:
            return logits
        

        # ppl_logits = self.pred(torch.cat((joint_repr, a_vec), dim=-1)).squeeze(1)
        

        if labels is not None:
            loss = self.loss_func(ppl_logits, labels)
            return loss
        return ppl_logits

# class Discriminator(nn.Module):
#     def __init__(self, vocab, d_v, d_word, d_model, dropout=0.2):
#         super().__init__()
#         n_layers = 2
#         n_heads = 6
#         attn_drop = dropout
#         self.d_v = d_v
#         self.d_model = d_model
#         self.v_in = nn.Sequential(nn.Linear(self.d_v, self.d_model),
#                                 nn.LeakyReLU(0.1),
#                                 nn.Dropout(attn_drop))
#         self.text_encoder = EncoderRNN(vocab=vocab, input_size=d_word, hidden_size=d_word, rnn_type="lstm", bidirectional=True, rnn_dropout=dropout)
#         self.ans_encoder = EncoderRNN(vocab=vocab, input_size=d_word, hidden_size=d_word, rnn_type="gru", bidirectional=False, rnn_dropout=dropout)

        
#         self.spatial_feats = nn.Sequential(nn.Linear(6, 300),
#                                             nn.LeakyReLU(0.1),
#                                             nn.Dropout(dropout))
#         self.vh_feats = nn.Sequential(nn.Linear(1, 300),
#                                         nn.LeakyReLU(0.1),
#                                         nn.Dropout(dropout))
#         self.v_feats = nn.Sequential(nn.Linear(d_v + 300, d_model),
#                                         nn.LeakyReLU(0.1),
#                                         nn.Dropout(dropout))
#         self.interaction_v_q = Aggregation(d_query=d_model, d_memory=d_word, d_model=d_word)
#         self.interaction_v_a = Aggregation(d_query=d_model, d_memory=d_word, d_model=d_word)
#         self.vis_emb = nn.Sequential(nn.Linear(self.d_model + 1 * d_word, self.d_model),
#                                         nn.LeakyReLU(0.1),
#                                         nn.Dropout())
#         self.pred = nn.Sequential(nn.Linear(self.d_model + d_word*2, self.d_model),
#                                 nn.Tanh(),
#                                 nn.Dropout(dropout),
#                                 nn.Linear(self.d_model, 1))
#         self.loss_func = nn.BCEWithLogitsLoss()
#         self.attn = Attention(query_size=d_word*2, memory_size=d_model, hidden_size=d_model, has_bias=True)

#         self.neg_attn = Attention(query_size=d_word*3, memory_size=d_model, hidden_size=d_model, has_bias=True)
    
#         self.q_net = FCNet([2*d_word, d_model])
#         self.v_net = FCNet([d_model, d_model])

#     def forward(self, visual_feats, visual_spatial_feats, visual_hints, question, question_mask, answer, labels=None):
#         spatial_feats = self.spatial_feats(visual_spatial_feats)
#         v_feats = torch.cat((visual_feats, spatial_feats), dim=2)
#         v_feats = self.v_feats(v_feats)
#         q_lens = question_mask.sum(1)
#         q_lens[q_lens == 0] = 1
#         q_feats, q_vec, _ = self.text_encoder(question)
#         q_vec = torch.cat((q_vec[0].squeeze(0), q_vec[1].squeeze(0)), dim=1)

#         a_feats, a_vec, a_mask = self.text_encoder(answer)
#         # a_vec = a_vec.squeeze(0)
#         a_vec = torch.cat((a_vec[0].squeeze(0), a_vec[1].squeeze(0)), dim=1)

#         pos_collect = []
#         neg_collect = []
#         # attn_mask = visual_feats.new_zeros((visual_feats.shape[0], 36))
#         # for i in range(visual_feats.shape[0]):
#         #     sub_v_feats = v_feats[i]
#         #     sub_vh = visual_hints[i]
#         #     pos = sub_v_feats[sub_vh==1]
#         #     if sub_vh.sum().item() == 0:
#         #         pos = sub_v_feats
#         #     attn_mask[i, :pos.shape[0]] = 1
#         #     pos_collect.append(pos)
#         #     # neg = sub_v_feats[sub_vh==0]
#         #     # neg_collect.append(neg)
#         # from torch.nn.utils.rnn import pad_sequence
#         # paded_pos = pad_sequence(pos_collect).transpose(0, 1)
#         # attn_mask = attn_mask[:, :paded_pos.shape[1]]

#         attn_res, _ = self.attn(query=q_vec, memory=v_feats, memory_mask=None)

#         q = self.q_net(q_vec)
#         v = self.v_net(attn_res)
#         joint_repr = q * v
        

#         ppl_logits = self.pred(torch.cat((joint_repr, a_vec), dim=-1)).squeeze(1)
        

#         if labels is not None:
#             loss = self.loss_func(ppl_logits, labels)
#             return loss
#         return ppl_logits

#         pass