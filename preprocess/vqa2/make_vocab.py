"""
    Meta-information:
        vocab_save_path = '/home/shiina/data/nips/vqa2/processed/vqa2_vocab_unique.json'
        threshold: 1
        Containing 16785 words
"""

import os, pickle
from data.build_vocab import VocabBuilder
from utils.text import tokenize


if __name__ == "__main__":
    # builder = VocabBuilder(vocab_thresh=1)
    # train_split_path = "/home/shiina/data/iclr_further/vqa2/split_data/train_split.pkl"
    # vocab_save_path = '/home/shiina/data/iclr_further/vqa2/vqa2_vocab.json'

    # with open(train_split_path, "rb") as f:
    #     train_split = pickle.load(f)
    # for qid, inst in train_split.items():
    #     ques_tokens = inst["question_tokens"]
    #     ans_tokens = inst['answer_tokens']
    #     builder.add2counters(ques_tokens)
    #     builder.add2vocabs(ans_tokens)

    # builder.finish()
    # builder.process()
    # builder.print()
    # builder.save(vocab_save_path)


    builder_unique = VocabBuilder(vocab_thresh=1)
    train_split_path = "/home/shiina/data/nips/vqa2/processed/train_split.pkl"
    vocab_save_path = '/home/shiina/data/nips/vqa2/processed/vqa2_vocab_unique.json'

    with open(train_split_path, "rb") as f:
        train_split = pickle.load(f)
    for inst in train_split:
        ques_tokens = inst["question_tokens"]
        ans_tokens = inst['answer_tokens']
        builder_unique.add2counters(ques_tokens)
        builder_unique.add2vocabs(ans_tokens)

    builder_unique.finish()
    builder_unique.process()
    builder_unique.print()
    builder_unique.save(vocab_save_path)
