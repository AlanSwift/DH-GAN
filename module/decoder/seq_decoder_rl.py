import torch
import torch.nn as nn
import torch.nn.functional as F
from data.vocab import Vocabulary
from module.submodule.attention import Attention


class RNNDecoderRL(nn.Module):
    def __init__(self, max_step, vocab: Vocabulary, dim_img, dim_ppl, dim_word, hidden_size, dropout, device,
                 fuse_strategy="average"):
        super(RNNDecoderRL, self).__init__()
        self.num_layers = 2
        self.hidden_size = hidden_size
        self.fuse_strategy = fuse_strategy
        self.vis_rnn = nn.LSTMCell(dim_img + dim_word, hidden_size)
        if self.fuse_strategy == "average":
            assert dim_img == dim_ppl
            self.lang_rnn = nn.LSTMCell(hidden_size + dim_img, hidden_size)
        elif self.fuse_strategy == "concat":
            self.lang_rnn = nn.LSTMCell(hidden_size + dim_img + dim_ppl, hidden_size)
        else:
            raise NotImplementedError()
        self.max_step = max_step
        self.vocab = vocab
        self.word_emb = nn.Sequential(nn.Embedding(len(vocab), dim_word),
                                      nn.ReLU(),
                                      nn.Dropout(dropout))
        self.device = device
        self.attn_pool = Attention(query_size=dim_word, memory_size=dim_img, hidden_size=dim_word,
                                   has_bias=True, dropout=dropout)
        self.attn_image = Attention(query_size=hidden_size, memory_size=dim_img, hidden_size=hidden_size,
                                    has_bias=True, dropout=dropout)
        self.attn_ppl = Attention(query_size=hidden_size, memory_size=dim_ppl, hidden_size=hidden_size,
                                  has_bias=True, dropout=dropout)
        self.project = nn.Linear(hidden_size, len(vocab))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, ppl_feats, answer_hint, question=None, sampling=False):
        batch_size = img_feats.shape[0]
        rnn_state = self.init_hidden(batch_size)
        decoder_in = torch.zeros(batch_size).fill_(self.vocab.word2idx[self.vocab.SYM_SOS]).long().to(self.device)

        img_pool, _ = self.attn_pool(query=answer_hint, memory=img_feats)
        decoder_out_collect = []
        use_teacher_forcing = question is not None
        decoding_tokens = []

        for idx in range(self.max_step):
            if all(decoder_in == self.vocab.word2idx[self.vocab.SYM_PAD]):
                break
            dec_emb = self.word_emb(decoder_in)
            decoder_out, rnn_state = self._decode_step(dec_input_emb=dec_emb, rnn_state=rnn_state, image_feats=img_feats, image_pool=img_pool,
                                                       ppl_feats=ppl_feats, answer_hint=answer_hint)
            decoder_out_logits = self.project(decoder_out)
            logprob = torch.log_softmax(decoder_out_logits, dim=-1)
            decoder_out_collect.append(logprob.unsqueeze(1))

            if use_teacher_forcing:
                decoder_in = question[:, idx]
            elif sampling:
                prob_distribution = torch.exp(logprob)
                decoder_in = torch.multinomial(prob_distribution, 1).squeeze(1)
            else:
                decoder_in = logprob.argmax(dim=-1)

            if sampling:
                decoding_tokens.append(decoder_in.unsqueeze(1))
            else:
                decoding_tokens.append(logprob.argmax(dim=-1).unsqueeze(1))
        decoder_ret = torch.cat(decoder_out_collect, dim=1)
        decoder_tokens = torch.cat(decoding_tokens, dim=1)
        return decoder_ret, decoder_tokens


    def _decode_step(self, dec_input_emb, rnn_state, image_feats, image_pool, ppl_feats, answer_hint):
        visual_lstm_input = torch.cat((image_pool, dec_input_emb), dim=-1)
        h_vis, c_vis = self.vis_rnn(visual_lstm_input, (rnn_state[0][0], rnn_state[1][0]))
        image_attn, _ = self.attn_image(query=h_vis, memory=image_feats)
        ppl_attn, _ = self.attn_ppl(query=h_vis, memory=ppl_feats)
        if self.fuse_strategy == "average":
            language_lstm_input = torch.cat((image_attn + ppl_attn, h_vis), dim=-1)
        elif self.fuse_strategy == "concat":
            language_lstm_input = torch.cat((image_attn, ppl_attn, h_vis), dim=-1)
        else:
            raise NotImplementedError()
        h_lang, c_lang = self.lang_rnn(language_lstm_input, (rnn_state[0][1], rnn_state[1][1]))
        rnn_results = self.dropout(h_lang)
        rnn_state = (torch.stack([h_vis, h_lang]), torch.stack([c_vis, c_lang]))
        return rnn_results, rnn_state

    def init_hidden(self, bsz):
        return (torch.zeros(self.num_layers, bsz, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, bsz, self.hidden_size).to(self.device))
