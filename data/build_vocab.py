from stanfordcorenlp import StanfordCoreNLP
from data.vocab import Vocabulary
import argparse
from collections import Counter


class VocabBuilder:
    def __init__(self, vocab_thresh=3):
        self.vocab = Vocabulary()
        self.counter = Counter()
        self.vocab.clear()
        self.counter.clear()
        self.finish_adding = False
        self.vocab_thresh = vocab_thresh

    def clear(self):
        self.vocab.clear()
        self.counter.clear()
        self.finish_adding = False

    def finish(self):
        self.finish_adding = True

    @classmethod
    def from_config(cls, args):
        return cls(args.vocab_thresh)

    def add2vocabs(self, token_list: [str]):
        for word in token_list:
            self.vocab.add_word(word)

    def add2counters(self, token_list):
        self.finish_adding = False
        self.counter.update(token_list)
        pass

    def process(self):
        assert self.finish_adding is True, "[Error]: The vocabulary is still adding"
        words = []
        words.extend([word for word, cnt in self.counter.items() if cnt >= self.vocab_thresh])
        words = list(set(words))
        self.add2vocabs(words)

    def save(self, output_path: str):
        self.vocab.save(output_path)

    def print(self):
            self.vocab.print()
            pass


