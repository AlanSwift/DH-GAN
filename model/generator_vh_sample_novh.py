import torch
import torch.nn as nn
import torch.nn.functional as F
from module.encoder import str2encoder
from module.decoder import str2decoder
from module.submodule.loss import Graph2seqLoss, VisualHintLoss, VisualHintLossBalancedALL, KDLoss, VisualHintLossFocal, VisualHintLossHinge
from module.encoder.ppl_encoder import VisualHintPredictor
from utils.gan_utils import extract_visual_hint_from_prob, sample_vh_from_prob


class Graph2seqGeneratorVhSamples(nn.Module):
    def __init__(self, hidden_size,  # cnn encoder config
                 text_encoder, vocab, dim_word,
                 graph_encoder, dim_ppl, dim_loc_feats, dim_visual_hint, topk,
                 seq_decoder, text_max_length,
                 dropout, device, ppl_num, gnn,
                 epsilon=0.75, temperature=1,
                 ):
        super(Graph2seqGeneratorVhSamples, self).__init__()
        self._build(hidden_size=hidden_size, dropout=dropout,
                    text_encoder=text_encoder, vocab=vocab, dim_word=dim_word,
                    graph_encoder=graph_encoder, dim_ppl=dim_ppl, dim_loc_feats=dim_loc_feats, dim_visual_hint=dim_visual_hint, topk=topk,
                    seq_decoder=seq_decoder, text_max_length=text_max_length, device=device, ppl_num=ppl_num, gnn=gnn, epsilon=epsilon)
        self.loss_criteria = Graph2seqLoss(vocab=vocab)
        self.loss_visual_hint = VisualHintLossFocal(alpha=4, gamma=2)
        self.loss_vh_hinge = VisualHintLossHinge(alpha=4, gamma=2)
        self.loss_l2 = nn.MSELoss()
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.ppl_in = nn.Linear(dim_ppl + 6, hidden_size)
        self.vh_prj = nn.Linear(1, 300)
        self.double_hints_ppl = nn.Sequential(nn.Linear(hidden_size + 300, hidden_size),
                                                nn.ReLU(),
                                                nn.Dropout(dropout))
        self.temperature = temperature

    def _build(self, text_encoder, vocab, dim_word,
               graph_encoder, dim_ppl, dim_loc_feats, dim_visual_hint, topk,
               seq_decoder, text_max_length,
               hidden_size, dropout, device, ppl_num, gnn, epsilon):

        self.text_encoder = str2encoder[text_encoder](vocab=vocab, input_size=dim_word, hidden_size=dim_word,
                                                      rnn_type="gru", rnn_dropout=dropout, device=device)
        # self.graph_encoder = str2encoder[graph_encoder](dim_ppl=dim_ppl, dim_loc_feats=dim_loc_feats, topk=topk,
        #         dim_visual_hint=dim_visual_hint, dim_answer=dim_word, hidden_size=hidden_size, dropout=dropout,
        #         ppl_num=ppl_num, gnn=gnn, epsilon=epsilon)
        self.vh_predictor = VisualHintPredictor(dim_ppl=hidden_size, dim_answer=dim_word, hidden_size=hidden_size, dropout=dropout)
        self.rnn_decoder = str2decoder[seq_decoder](max_step=text_max_length, vocab=vocab, dim_img=hidden_size,
                dim_word=dim_word, hidden_size=hidden_size, dropout=dropout, device=device, dim_ppl=hidden_size,
                                                    fuse_strategy="image")

    @classmethod
    def from_opts(cls, args, vocab, device):
        return cls(hidden_size=args.hidden_size,
                   text_encoder=args.text_encoder, vocab=vocab, dim_word=args.word_dim,
                   graph_encoder=args.graph_encoder, dim_ppl=args.proposal_dim, dim_loc_feats=args.loc_feats_dim,
                   dim_visual_hint=args.visual_hint_dim, topk=args.topk,
                   seq_decoder=args.seq_decoder, text_max_length=args.text_max_length,
                   dropout=args.dropout, device=device, ppl_num=args.ppl_num, gnn=args.gnn,
                   epsilon=args.epsilon, temperature=args.temperature)

    def forward(self, ppl_feats, ppl_info, question=None, answer=None, sampling_procedure=False):
        if sampling_procedure:
            return self.sample(ppl_feats=ppl_feats, ppl_info=ppl_info, question=question, answer=answer)

        answer_hint, answer_vec, answer_mask = self.text_encoder(answer, answer.sum(1))
        answer_hint = answer_hint.transpose(0, 1)
        # answer_hint, _ = torch.max(answer_hint, dim=1)
        # , _ = torch.max(answer_hint, dim=1)
        answer_vec = answer_vec.transpose(0, 1).squeeze(1).contiguous()

        ppl = torch.cat((ppl_feats, ppl_info), dim=-1)
        ppl = self.ppl_in(ppl)

        vh_pred = self.vh_predictor(ppl, answer_vec)
        vh_pred_label = extract_visual_hint_from_prob(vh_pred, threshold=0.1).unsqueeze(2)
        # vh_pred_label = torch.softmax(vh_pred, dim=1) >= 0.05
        # vh_pred_label = sample_vh_from_prob(vh_pred.squeeze(2)).unsqueeze(2)
        vh_feats = self.vh_prj(vh_pred_label.float().detach())
        ppl_with_vh = torch.cat((ppl, vh_feats), dim=2)
        ppl = self.double_hints_ppl(ppl_with_vh)

        decoder_mask = vh_pred_label.squeeze(2).detach()
        # for i in range(decoder_mask.shape[0]):
        #     if decoder_mask[i].sum().item() == 0:
        #         decoder_mask[i] = 1.

        need_mask = decoder_mask.sum(1) == 0
        decoder_mask = decoder_mask.masked_fill(need_mask.unsqueeze(1), 1)

        logits_student = self.rnn_decoder(img_feats=ppl, ppl_feats=None, answer_hint=answer_vec, question=question, ppl_mask=decoder_mask) # , ppl_mask=pred

        step_cnt = logits_student.shape[1]
        if question is not None:
            loss_student, _ = self.loss_criteria(prob=logits_student, gt=question[:, 0:step_cnt].contiguous())
            return loss_student, vh_pred
        else:
            return logits_student, vh_pred_label.squeeze(2)

    def predict_vh_logits(self, ppl_feats, ppl_info, answer=None):
        answer_hint, answer_vec, answer_mask = self.text_encoder(answer, answer.sum(1))
        answer_hint = answer_hint.transpose(0, 1)
        # answer_hint, _ = torch.max(answer_hint, dim=1)
        # , _ = torch.max(answer_hint, dim=1)
        answer_vec = answer_vec.transpose(0, 1).squeeze(1).contiguous()

        ppl = torch.cat((ppl_feats, ppl_info), dim=-1)
        ppl = self.ppl_in(ppl)

        vh_pred = self.vh_predictor(ppl, answer_vec)
        return vh_pred
    
    def sample(self, ppl_feats, ppl_info, question=None, answer=None):
        answer_hint, answer_vec, answer_mask = self.text_encoder(answer, answer.sum(1))
        answer_hint = answer_hint.transpose(0, 1)
        # answer_hint, _ = torch.max(answer_hint, dim=1)
        # , _ = torch.max(answer_hint, dim=1)
        answer_vec = answer_vec.transpose(0, 1).squeeze(1).contiguous()

        ppl = torch.cat((ppl_feats, ppl_info), dim=-1)
        ppl = self.ppl_in(ppl)

        vh_pred = self.vh_predictor(ppl, answer_vec)
        vh_pred_label = extract_visual_hint_from_prob(vh_pred, threshold=0.1).unsqueeze(2)

        # vh_pred_label = sample_vh_from_prob(vh_pred, temperature=self.temperature).unsqueeze(2)
        # vh_pred_label = torch.softmax(vh_pred, dim=1) >= 0.05
        vh_feats = self.vh_prj(vh_pred_label.float())
        ppl_with_vh = torch.cat((ppl, vh_feats), dim=2)
        ppl = self.double_hints_ppl(ppl_with_vh)

        logits, sampled_results = self.rnn_decoder(img_feats=ppl, ppl_feats=None, answer_hint=answer_vec, question=question, ppl_mask=None, sampling=True) # , ppl_mask=pred

        return logits, sampled_results.detach(), vh_pred, vh_pred_label.squeeze(2).detach()
