import torch
import torch.nn as nn
import torch.nn.functional as F
from data.vocab import Vocabulary
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import init

def mean_pool_with_mask(x: torch.Tensor, length):
    xx = x.sum(1)
    return xx / length.unsqueeze(1)


class TextEncoder(nn.Module):
    def __init__(self, vocab: Vocabulary, dim_word, dim_out, encoder_style, num_layer=1, bidirectional=False, dropout=0.2):
        super(TextEncoder, self).__init__()
        self.word_emb = nn.Sequential(nn.Embedding(len(vocab), dim_word),
                                      nn.ReLU(),
                                      nn.Dropout(dropout))
        self.encoder_style = encoder_style
        self.vocab = vocab
        if encoder_style == "LSTM":
            self.encoder = nn.LSTM(input_size=dim_word, hidden_size=dim_out, num_layers=num_layer, bidirectional=bidirectional)
        elif encoder_style == "mean":
            self.encoder = mean_pool_with_mask

    def forward(self, text):
        text_emb = self.word_emb(text)
        pad_idx = self.vocab.word2idx[self.vocab.SYM_PAD]
        msk = text != pad_idx
        length = msk.float().sum(1)
        max_length = torch.max(length)


        text_emb_msk = text_emb.masked_fill(mask=(1 - msk.unsqueeze(2).float()).bool(), value=-1e8)
        max_pool, _ = torch.max(text_emb_msk, dim=1)

        # if self.encoder_style == "mean":
        #     text_emb = self.encoder(text_emb, (text != pad_idx).float().sum(1))
        return max_pool, text_emb[:, 0:max_length.long().item(), :], msk


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask


class EncoderRNNDifferentiable(nn.Module):
    def __init__(self, vocab, input_size, hidden_size, \
        bidirectional=False, num_layers=1, rnn_type='lstm', rnn_dropout=None, device=None):
        super(EncoderRNNDifferentiable, self).__init__()
        self.word_matrix = nn.Parameter(torch.Tensor(len(vocab), input_size))
       
        # self.word_emb = nn.Embedding(len(vocab), input_size, padding_idx=self.vocab.word2idx[self.vocab.SYM_PAD])
        self.vocab = vocab
        self.reset_parameters()

        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))
        if bidirectional:
            print('[ Using {}-layer bidirectional {} encoder ]'.format(num_layers, rnn_type))
        else:
            print('[ Using {}-layer {} encoder ]'.format(num_layers, rnn_type))
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)

    def reset_parameters(self) -> None:
        init.normal_(self.word_matrix)
        with torch.no_grad():
            self.word_matrix[self.vocab.word2idx[self.vocab.SYM_PAD]].fill_(0)

    def forward(self, x, x_len):
        """x: [batch_size * max_length * emb_dim]
           x_len: [batch_size]
        """
        device = x.device

        max_length = torch.max(x_len).long().item()
        x = x[:, :max_length]

        x = torch.matmul(x, self.word_matrix)

        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size).to(device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size).to(device)
            packed_h, (packed_h_t, packed_c_t) = self.model(x, (h0, c0))
        else:
            packed_h, packed_h_t = self.model(x, h0)

        if self.num_directions == 2:
            packed_h_t = torch.cat((packed_h_t[-1], packed_h_t[-2]), 1)
            if self.rnn_type == 'lstm':
                packed_c_t = torch.cat((packed_c_t[-1], packed_c_t[-2]), 1)
        else:
            packed_h_t = packed_h_t[-1]
            if self.rnn_type == 'lstm':
                packed_c_t = packed_c_t[-1]

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)
        restore_hh = hh[inverse_indx]
        restore_hh = self.rnn_dropout(restore_hh)
        restore_hh = restore_hh.transpose(0, 1) # [max_length, batch_size, emb_dim]

        restore_packed_h_t = packed_h_t[inverse_indx]
        restore_packed_h_t = self.rnn_dropout(restore_packed_h_t)
        restore_packed_h_t = restore_packed_h_t.unsqueeze(0) # [1, batch_size, emb_dim]

        if self.rnn_type == 'lstm':
            restore_packed_c_t = packed_c_t[inverse_indx]
            restore_packed_c_t = self.rnn_dropout(restore_packed_c_t)
            restore_packed_c_t = restore_packed_c_t.unsqueeze(0) # [1, batch_size, emb_dim]
            rnn_state_t = (restore_packed_h_t, restore_packed_c_t)
        else:
            rnn_state_t = restore_packed_h_t
        return restore_hh, rnn_state_t

from utils.bert_utils import *

class BertEmbedding(nn.Module):
    """Bert embedding class.

    Parameters
    ----------
    name : str, optional
        BERT model name, default: ``'bert-base-uncased'``.
    max_seq_len : int, optional
        Maximal sequence length, default: ``500``.
    doc_stride : int, optional
        Chunking stride, default: ``250``.
    fix_emb : boolean, optional
        Specify whether to fix pretrained BERT embeddings, default: ``True``.
    lower_case : boolean, optional
        Specify whether to use lower case, default: ``True``.

    """
    def __init__(self,
                name='bert-base-uncased',
                max_seq_len=500,
                doc_stride=250,
                fix_emb=True,
                lower_case=True):
        super(BertEmbedding, self).__init__()
        self.bert_max_seq_len = max_seq_len
        self.bert_doc_stride = doc_stride
        self.fix_emb = fix_emb

        from transformers import BertModel
        from transformers import BertTokenizer
        print('[ Using pretrained BERT embeddings ]')
        self.bert_tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=lower_case)
        self.bert_model = BertModel.from_pretrained(name)
        if fix_emb:
            print('[ Fix BERT layers ]')
            self.bert_model.eval()
            for param in self.bert_model.parameters():
                param.requires_grad = False
        else:
            print('[ Finetune BERT layers ]')
            self.bert_model.train()

        # compute weighted average over BERT layers
        self.logits_bert_layers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, self.bert_model.config.num_hidden_layers)))


    def forward(self, raw_text_data):
        """Compute BERT embeddings for each word in text.

        Parameters
        ----------
        raw_text_data : list
            The raw text input data. Example: [['what', 'is', 'bert'], ['how', 'to', 'use', 'bert']].

        Returns
        -------
        torch.Tensor
            BERT embedding matrix.
        """
        bert_features = []
        max_d_len = 0
        for text in raw_text_data:
            bert_features.append(convert_text_to_bert_features(text, self.bert_tokenizer, self.bert_max_seq_len, self.bert_doc_stride))
            max_d_len = max(max_d_len, len(text))

        max_bert_d_num_chunks = max([len(ex_bert_d) for ex_bert_d in bert_features])
        max_bert_d_len = max([len(bert_d.input_ids) for ex_bert_d in bert_features for bert_d in ex_bert_d])
        bert_xd = torch.LongTensor(len(raw_text_data), max_bert_d_num_chunks, max_bert_d_len).fill_(0)
        bert_xd_mask = torch.LongTensor(len(raw_text_data), max_bert_d_num_chunks, max_bert_d_len).fill_(0)
        for i, ex_bert_d in enumerate(bert_features): # Example level
            for j, bert_d in enumerate(ex_bert_d): # Chunk level
                bert_xd[i, j, :len(bert_d.input_ids)].copy_(torch.LongTensor(bert_d.input_ids))
                bert_xd_mask[i, j, :len(bert_d.input_mask)].copy_(torch.LongTensor(bert_d.input_mask))

        bert_xd = bert_xd.to(self.bert_model.device)
        bert_xd_mask = bert_xd_mask.to(self.bert_model.device)

        encoder_outputs = self.bert_model(bert_xd.view(-1, bert_xd.size(-1)),
                                        token_type_ids=None,
                                        attention_mask=bert_xd_mask.view(-1, bert_xd_mask.size(-1)),
                                        output_hidden_states=True,
                                        return_dict=True)
        all_encoder_layers = encoder_outputs['hidden_states'][1:] # The first one is the input embedding
        all_encoder_layers = torch.stack([x.view(bert_xd.shape + (-1,)) for x in all_encoder_layers], 0)
        bert_xd_f = extract_bert_hidden_states(all_encoder_layers, max_d_len, bert_features, weighted_avg=True)

        weights_bert_layers = torch.softmax(self.logits_bert_layers, dim=-1)
        bert_xd_f = torch.mm(weights_bert_layers, bert_xd_f.view(bert_xd_f.size(0), -1)).view(bert_xd_f.shape[1:])

        return bert_xd_f
