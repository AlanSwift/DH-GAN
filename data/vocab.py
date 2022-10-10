import json
from stanfordcorenlp import StanfordCoreNLP
import torch


class Vocabulary(object):
    SYM_UNK = 'UNK'  # Unknown word.
    SYM_PAD = 'PAD'
    SYM_SOS = 'SOS'

    def __init__(self):
        # Init mappings between words and ids
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word(self.SYM_UNK)
        self.add_word(self.SYM_SOS)
        self.add_word(self.SYM_PAD)

    def clear(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word(self.SYM_UNK)
        self.add_word(self.SYM_SOS)
        self.add_word(self.SYM_PAD)

    def add_word(self, word: str):
        assert type(word) is str, "[Error]: word should be str type while adding words"
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def print(self):
        print("----- Vocabulary Statistics -----")
        print("Contains: {} words".format(len(self.word2idx)))

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.SYM_UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save(self, location):
        with open(location, 'w') as f:
            json.dump({'word2idx': self.word2idx,
                       'idx2word': self.idx2word,
                       'idx': self.idx}, f)

    @classmethod
    def load(cls, location):
        with open(location, 'r') as f:
            data = json.load(f)
            ret = cls()
            ret.word2idx = data['word2idx']
            ret.idx2word = data['idx2word']
            ret.idx = data['idx']
            # if cls.SYM_PAD not in ret.word2idx:
            #     ret.add_word(cls.SYM_PAD)
            # if cls.SYM_SOS not in ret.word2idx:
            #     ret.add_word(cls.SYM_SOS)
            return ret

    def convert_tokens(self, token_list: list, token_rep: list = None) -> list:
        ret = []
        if token_rep is not None:
            for token, token_rep in zip(token_list, token_rep):
                if token not in self.word2idx:
                    if token_rep not in self.word2idx:
                        ret.append(self.word2idx[self.SYM_UNK])
                    else:
                        ret.append(self.word2idx[token_rep])
                else:
                    ret.append(self.word2idx[token])
        else:
            ret.extend([self(token) for token in token_list])
        return ret

    def convert_ids(self, probs):
        assert isinstance(probs, torch.Tensor)
        assert len(probs.shape) == 2
        sentence_collect = []
        for i in range(probs.shape[0]):
            sentence_ids = probs[i]
            token_collect = []
            for j in range(len(sentence_ids)):
                token = self.idx2word[str(sentence_ids[j].item())]
                if token == self.SYM_PAD:
                    break
                token_collect.append(token)
            sentence_collect.append(" ".join(token_collect))
        return sentence_collect

