"""
    Meta-information:
        raw_train_split_path: "/home/shiina/data/nips/vqa2/annotation/raw/train_instances.pkl"
        raw_val_split_path: "/home/shiina/data/nips/vqa2/annotation/raw/val_instances.pkl"
"""


import os
import pickle
import json
import copy
from tqdm import tqdm

def merge(question, answers, ppl_dir, resnet_dir):
    ans = {inst["question_id"]: inst for inst in answers}
    ques = {inst["question_id"]: inst for inst in question}
    assert len(ans.keys()) == len(ques.keys())
    ret = {}
    for qid in ans.keys():
        ppl_path = os.path.join(ppl_dir, str(int(qid)//1000) + ".pkl")
        if not os.path.exists(ppl_path):
            print(qid)
            exit(0)
        
        resnet_path = os.path.join(resnet_dir, str(int(qid)//1000) + ".npy")
        if not os.path.exists(resnet_path):
            print(qid)
            exit(0)

        ans_inst = ans[qid]
        ques_inst = ques[qid]
        tmp = copy.deepcopy(ans_inst)
        tmp.update(ques_inst)
        ret[qid] = tmp
        ret[qid]["ppl_path"] = ppl_path
        ret[qid]["resnet_path"] = resnet_path
    return ret

def detect(image_id, question, answer, split):
    if split == "train":
        pic_root = "/home/shiina/data/coco2014/train2014/pic"
        image_name = "COCO_train2014_{}.jpg".format(str(image_id).zfill(12))
        image_path = os.path.join(pic_root, image_name)

    else:
        pic_root = "/home/shiina/data/coco2014/val2014/pic"
        image_name = "COCO_val2014_{}.jpg".format(str(image_id).zfill(12))
        image_path = os.path.join(pic_root, image_name)

    convert_success = 0

    try:
        statement = nlp_tool.convert(question, answer, verbase=0)
        convert_success = 1
    except:
        statement = question + answer + "."
        statement = statement.lower()

    pred_bbox, conf = model.predict(image_path, statement)
    return pred_bbox, conf, convert_success

if __name__ == "__main__":

    train_answer_file = '/home/shiina/data/vqa2/v2_mscoco_train2014_annotations.json'
    ppl_path = "/home/shiina/data/nips/vqa2/features/proposals/train"
    resnet_dir = "/home/shiina/data/iclr_further/img_resnet_feats/train"
    train_answers = json.load(open(train_answer_file))['annotations']

    train_question_file = '/home/shiina/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json'
    train_questions = json.load(open(train_question_file))['questions']

    train_data = merge(train_questions, train_answers, ppl_path, resnet_dir)

    train_annotation_save_path = "/home/shiina/data/nips/vqa2/annotation/raw"
    os.makedirs(train_annotation_save_path, exist_ok=True)

    with open(os.path.join(train_annotation_save_path, "train_instances.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    print("---------train split--------")
    print("Statistics: all:{}".format(len(train_data)))

    val_answer_file = '/home/shiina/data/vqa2/v2_mscoco_val2014_annotations.json'
    val_answers = json.load(open(val_answer_file))['annotations']

    val_question_file = '/home/shiina/data/vqa2/v2_OpenEnded_mscoco_val2014_questions.json'
    val_questions = json.load(open(val_question_file))['questions']

    ppl_path = "/home/shiina/data/nips/vqa2/features/proposals/val"
    resnet_dir = "/home/shiina/data/iclr_further/img_resnet_feats/val"

    val_data = merge(val_questions, val_answers, ppl_path, resnet_dir)
    val_annotation_save_path = "/home/shiina/data/nips/vqa2/annotation/raw"

    with open(os.path.join(val_annotation_save_path, "val_instances.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    print("------val split--------")
    print("Statistics: all:{}".format(len(val_data)))