import pickle

import torch, copy
import torchtext.vocab as vocab


class Annotator:
    def __init__(self) -> None:
        
        glove_text = vocab.GloVe(name='6B', dim=300)
        print(glove_text["animal"].shape)
        pass

    def annotate(self, instances):
        for q_id, inst in instances.items():
            print(inst)
            question = inst["question"].strip().lower()
            answer = inst["multiple_choice_answer"].strip().lower()
            ppl_path = inst["ppl_path"]
            with open(ppl_path, "rb") as f:
                ppl = pickle.load(f)
            object_id = ppl["object_id"]
            print(ppl)
            exit(0)
        pass


if __name__ == "__main__":

    annotator = Annotator()

    train_input_instances_path = "/home/shiina/data/nips/vqa2/annotation/raw/train_instances.pkl"
    with open(train_input_instances_path, "rb") as f:
        train_instances = pickle.load(f)
    annotator.annotate(train_instances)


    val_input_instances = "/home/shiina/data/nips/vqa2/annotation/raw/val_instances.pkl"
    pass