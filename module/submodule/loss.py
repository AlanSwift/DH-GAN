import torch
import torch.nn as nn
from data.vocab import Vocabulary
import torch.nn.functional as F


class Graph2seqLoss(nn.Module):
    def __init__(self, vocab: Vocabulary):
        super(Graph2seqLoss, self).__init__()
        self.vocab = vocab

    def forward(self, prob, gt, reduction=True):
        batch_size = gt.shape[0]
        step = gt.shape[1]
        assert len(prob.shape) == 3
        assert len(gt.shape) == 2
        assert prob.shape[0:2] == gt.shape[0:2]
        mask = 1 - gt.data.eq(self.vocab.word2idx[self.vocab.SYM_PAD]).float()
        pad = mask.data.new(gt.shape[0], 1).fill_(1)
        mask = torch.cat((pad, mask[:, :-1]), dim=1)

        prob_select = torch.gather(prob.view(batch_size*step, -1), 1, gt.view(-1, 1))
        cnt = 0
        if reduction:
            prob_select_masked = - torch.masked_select(prob_select, mask.view(-1,1).bool())
            loss = torch.mean(prob_select_masked)
        else:
            prob_select = prob_select.view_as(gt)
            prob_select.masked_fill_(mask=(1 - mask).bool(), value=0)
            loss = - torch.sum(prob_select, dim=1)
            cnt = torch.sum(mask)
        return loss, cnt

class VisualHintLoss(nn.Module):
    def __init__(self):
        super(VisualHintLoss, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, pred, gt):
        return self.loss_func(pred.squeeze(2), gt)

class VisualHintLossBalanced(nn.Module):
    def __init__(self):
        super(VisualHintLossBalanced, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, pred, gt):
        log_p = F.log_softmax(pred, dim=-1)
        loss = gt * log_p
        return -torch.mean(loss)

class VisualHintLossBalancedALL(nn.Module):
    def __init__(self):
        super(VisualHintLossBalancedALL, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, pred, gt):
        return F.binary_cross_entropy_with_logits(pred, gt)

        prob = F.logsigmoid(pred)
        loss_p = prob * gt
        cnt_p = torch.sum(gt)
        neg_prob = torch.log(1 - F.sigmoid(pred) + 1e-9)
        loss_n = (1 - gt) * neg_prob
        cnt_n = torch.sum(1 - gt)

        # loss = torch.mean(loss_p + loss_n)
        # loss = torch.sum(loss_p) / cnt_p + torch.sum(loss_n) / cnt_n
        loss = torch.sum(loss_p) / cnt_p + torch.sum(loss_n) / cnt_n
        # loss /= pred.shape[0]
        return -loss


class VisualHintLossBalancedLine(nn.Module):
    def __init__(self):
        super(VisualHintLossBalancedLine, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, pred, gt):
        loss = self.loss_func(pred, gt.long())
        return -loss



class VisualHintLossHinge(nn.Module):
    def __init__(self, gamma, alpha):
        super(VisualHintLossHinge, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.small = 1e-31

    def forward(self, pred, gt):
        prob = F.softmax(pred, dim=-1)
        log_prob = torch.log(prob + self.small)

        tmp = prob[:, :, 0] - 0.5
        tmp = tmp.masked_fill(tmp <= 0, 0)
        loss_p = log_prob[:, :, 1] * gt * torch.pow(tmp, self.gamma)

        tmp = prob[:, :, 1] - 0.5
        tmp = tmp.masked_fill(tmp <= 0, 0)
        loss_n = log_prob[:, :, 0] * (1 - gt) * torch.pow(tmp, self.gamma)

        cnt_p = torch.sum(gt)
        # loss_n = (1 - gt) * prob[:, :, 0]
        cnt_n = torch.sum(1 - gt)


        loss = torch.sum(loss_p) / cnt_p + torch.sum(loss_n) / cnt_n
        return -loss * self.alpha


class VisualHintLossFocal(nn.Module):
    def __init__(self, gamma, alpha):
        super(VisualHintLossFocal, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.small = 1e-8

    def forward(self, pred, gt):
        prob = F.sigmoid(pred)
        log_prob = torch.log(prob + self.small)
        log_prob_neg = torch.log(1 - prob + self.small)

        loss_p = log_prob * gt * torch.pow(1 - prob + self.small, self.gamma)

        loss_n = log_prob_neg * (1 - gt) * torch.pow(prob + self.small, self.gamma)

        # prob = F.log_softmax(pred, dim=-1)
        # loss_p = prob[:, :, 1] * gt
        cnt_p = torch.sum(gt)
        # loss_n = (1 - gt) * prob[:, :, 0]
        cnt_n = torch.sum(1 - gt)

        # loss = torch.mean(loss_p + loss_n)
        # loss = torch.sum(loss_p) / cnt_p + torch.sum(loss_n) / cnt_n
        # loss = torch.sum(loss_p) / cnt_p + torch.sum(loss_n) / cnt_n
        loss = torch.sum(loss_p) + torch.sum(loss_n)
        return -loss * self.alpha / gt.shape[0]






class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.kd_loss = nn.KLDivLoss()

    def forward(self, teacher_map, student_map):
        l2_loss = self.l2_loss(teacher_map, student_map)
        kd_loss = self.kd_loss(teacher_map, student_map)
        return l2_loss, kd_loss