import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from data.vocab import Vocabulary
from utils.image import RandomCrop


class VQA2Dataset(data.Dataset):
    def __init__(self, split_dic_path, vocab_path,
                 prop_thresh=0, pad_length=20, verbase=1, ppl_num=36, split="train", ratio=1, answer_dict_path=None):
        super(VQA2Dataset, self).__init__()
        with open(split_dic_path, "rb") as f:
            self.dic = pickle.load(f)
        self.split = split
        self.ratio = ratio
        if answer_dict_path:
            with open(answer_dict_path , "rb") as f:
                self.answer2id = pickle.load(f)
        #     self.answer2id = {ans: ans_id for ans_id, ans in enumerate(answer_dict_list)}
        #     self.answer2id["UNK"] = len(answer_dict_list)
        #     print("Answer number", len(answer_dict_list) + 1) # 16367
        else:
            self.answer2id = None
        self.vocab = Vocabulary.load(vocab_path)
        # self.Resize = transforms.Resize((image_size, image_size))
        # self.rand_crop = RandomCrop(image_crop_size)

        # self.img_process = transforms.Compose([transforms.ToTensor(),
        #                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.prop_thresh = prop_thresh
        self.pad_length = pad_length
        self.verbase = verbase
        self.ppl_num = ppl_num

    def __getitem__(self, item):
        inst = self.dic[item]
        if self.verbase > 0:
            print(inst)

        image_path = inst["resnet_path"]
        question_idx = inst["question_id"]

        # # load resnet features
        # img_features = np.load(image_path)


        # load proposals
        with open(inst['ppl_path'], "rb") as f:
            ppl_json = pickle.load(f)

        ppl_spatial_info = ppl_json["spatial_features"]

        ppl_features = ppl_json["features"]

        # visual hint
        visual_hint = np.array(inst["visual_hint"])
        

        # convert question to id
        question = inst["question_tokens"]
        question_lemma = inst["question_lemmas"]
        question_ids = self.vocab.convert_tokens(question, question_lemma)
        question_ids = np.array(question_ids)
        question_str = inst["question"].lower().replace("?", "")

        answer = inst["answer_tokens"]
        answer_lemma = inst["answer_lemmas"]
        answer_ids = self.vocab.convert_tokens(answer, answer_lemma)
        answer_ids = np.array(answer_ids)
        ans_str = inst["multiple_choice_answer"]
        if self.answer2id:
            answer_idx = self.answer2id.get(ans_str, self.answer2id["UNK"])
        else:
            answer_idx = None


        box_feats = ppl_features[:self.ppl_num, :]
        box_infos = ppl_spatial_info[:self.ppl_num, :]
        visual_hint = visual_hint[:self.ppl_num]
        if box_feats.shape[0] == 35:
            print(box_feats.shape, "--------")


        if self.verbase > 0:
            # print("image", img_features.shape)
            print("box_feats", box_feats.shape, type(box_feats))
            print("box_infos", box_infos.shape, type(box_infos))
            print("question ids", question_ids.shape, type(question_ids))
            print("answer_ids", answer_ids.shape, type(answer_ids))
        # vectorize
        def pad(x, length=self.pad_length):
            assert len(x.shape) == 1
            assert isinstance(x, torch.Tensor)
            pad_len = length - x.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len).fill_(self.vocab(self.vocab.SYM_PAD))
                x = torch.cat((x, pad.long()), dim=0)
            elif pad_len <= 0:
                x = x[:length]
            return x
        question = torch.from_numpy(question_ids)
        question = pad(question)
        answer = torch.from_numpy(answer_ids)
        answer = pad(answer)


        if self.answer2id:
            answer_idx_tensor = torch.Tensor([answer_idx])
            return box_feats, box_infos, visual_hint, question, answer, question_str, question_idx, answer_idx_tensor

        else:
            return box_feats, box_infos, visual_hint, question, answer, question_str, question_idx
 
        img, box_feats, box_info, question, answer, visual_hint = self.vectorize(img=img, box_feats=box_feats,
                                                                                 box_infos=box_infos,
                                                                                 box_cls=box_cls, question=question_ids,
                                                                                 answer=answer_ids,
                                                                                 visual_hint=visual_hint)
        return img, box_feats, box_info, visual_hint, question, answer, answer_gt_idx, question_str, question_idx

    def __len__(self):
        return int(len(self.dic)*self.ratio)

if __name__ == "__main__":
    dataset = VQA2Dataset(split_dic_path="/home/shiina/data/nips/vqa2/processed/train_split.pkl",
                          vocab_path='/home/shiina/data/nips/vqa2/processed/vqa2_vocab_unique.json',
                          split="train", verbase=0)
    ds_loader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=1)
    for data in ds_loader:
        print("------------------")
        img, box_feats, box_info, visual_hint, question, answer, question_str, answer_idx = data
        print(img.shape, "img1111")
        print(box_feats.shape, "box_feats1111")
        print(box_info.shape, "box_info1111")
        print(visual_hint.shape, "visual_hint_1111")
        print(question.shape, "question111")
        print(answer.shape, "answer11111")
        print(answer_idx.shape, "answeridx")
        print(answer_idx)
        print(question_str)
        exit(0)
