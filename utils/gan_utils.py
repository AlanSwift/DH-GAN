import torch
import torch.nn as nn
import torch.nn.functional as F


# def extract_visual_hint_from_prob(prob, MAX_PPL=4):
#     if len(prob.shape) == 3:
#         prob = prob.squeeze(2)
#     ret = prob.new_zeros((prob.shape[0], prob.shape[1]))
#     for i in range(prob.shape[0]):
#         can = prob[i] >= 0.5
#         if can.sum().item() > MAX_PPL:
#             _, slides = prob[i].topk(MAX_PPL)
#             ret[i, slides] = 1
#         else:
#             ret[i] = can
#     return ret

def extract_visual_hint_from_prob(prob, threshold=0.1):
    assert len(prob.shape) == 2
    scores = torch.sigmoid(prob)
    _ = F.tanh(scores)
    labels_hard = scores >= 0.5
    ret = _ + labels_hard.float().detach() - _.detach()
    return ret

def sample_vh_from_prob(logits, temperature=1):
    prob = torch.sigmoid(torch.clamp(logits / temperature, 1e-8, 5))
    prob_neg = 1 - prob
    scores = torch.cat((prob_neg.unsqueeze(2), prob.unsqueeze(2)), dim=2)
    try:
        sampled = torch.multinomial(scores.view(-1, 2), num_samples=1)
    except:
        print(logits)
        print(scores)
        exit(0)
    return sampled.view(logits.shape[0], logits.shape[1])
    
    return ret

loss_func = nn.NLLLoss(ignore_index=2, reduce=False)



class Graph2seqLoss(nn.Module):
    def __init__(self):
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


def pg_loss(prob, gt, reward):
    batch_size = gt.shape[0]
    step = gt.shape[1]
    assert len(prob.shape) == 3
    assert len(gt.shape) == 2
    assert prob.shape[0:2] == gt.shape[0:2]
    mask = 1 - gt.data.eq(2).float()
    pad = mask.data.new(gt.shape[0], 1).fill_(1)
    mask = torch.cat((pad, mask[:, :-1]), dim=1)

    prob_select = torch.gather(prob.view(batch_size*step, -1), 1, gt.view(-1, 1))
    
    prob_select = prob_select.view_as(gt)
    prob_select.masked_fill_(mask=(1 - mask).bool(), value=0)
    loss = - torch.sum(prob_select*reward.unsqueeze(1)) / prob_select.shape[0]
    return loss







    global loss_func
    
    loss = F.nll_loss(log_prob.transpose(1, 2), target, ignore_index=2, reduction="none")

    loss_with_reward = torch.mul(loss, reward.unsqueeze(1))
    return loss_with_reward.sum() / target.shape[0]
    print(loss.shape)
    exit(0)

    out = - log_prob.gather(index=target.unsqueeze(2), dim=2).squeeze(2)
    out_with_reward = torch.mul(-out, reward.unsqueeze(1))
    return out_with_reward.sum()/ target.shape[0]
    print(out.shape)
    exit(0)
    loss = 0
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            loss += -log_prob[i][j][target.data[i][j]]*reward[i]     # log(P(y_t|Y_1:Y_{t-1})) * Q
            if target[i][j].item() == 2:
                break

    return loss/target.shape[0]
    pass

def vh_pg_loss(vh_logits, vh_target, reward):
    prob = - torch.log_softmax(vh_logits.squeeze(2), dim=1)
    prob_pos = torch.mul(prob, vh_target.float())
    prob_pos_with_reward = torch.mul(prob_pos, reward.unsqueeze(1))
    loss = torch.sum(prob_pos_with_reward) / prob.shape[0]
    return loss