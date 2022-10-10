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

def merge(question_path, answers_path, image_ids_path, ppl_dir, resnet_dir):
    with open(question_path, "r") as f:
        ques_lines = f.readlines()
    
    with open(answers_path, "r") as f:
        ans_lines = f.readlines()

    with open(image_ids_path, "r") as f:
        image_id_lines = f.readlines()
    
    assert len(ques_lines) == len(ans_lines)
    assert len(ques_lines) == len(image_id_lines)

    ret = {}

    for q, a, img_id in zip(ques_lines, ans_lines, image_id_lines):
        if q.strip() == "":
            continue
        qid = int(img_id) * 1000
        while qid in ret:
            qid += 1
        
        ppl_path = os.path.join(ppl_dir, str(int(qid)//1000) + ".pkl")
        if not os.path.exists(ppl_path):
            print(qid, "-----")
            exit(0)
        
        resnet_path = os.path.join(resnet_dir, str(int(qid)//1000) + ".npy")
        if not os.path.exists(resnet_path):
            print(qid, "======")
            exit(0)
        item = {
            "ppl_path": ppl_path,
            "resnet_path": resnet_path,
            "question": q,
            "multiple_choice_answer": a,
            "question_id": qid
        }
        ret[qid] = item
        
    return ret



if __name__ == "__main__":

    train_answer_file = "/home/shiina/data/cocoqa/train/answers.txt"
    train_question_file = "/home/shiina/data/cocoqa/train/questions.txt"
    train_images_file = "/home/shiina/data/cocoqa/train/img_ids.txt"
    ppl_path = "/home/shiina/data/nips/vqa2/features/proposals/train"
    resnet_dir = "/home/shiina/data/iclr_further/img_resnet_feats/train"

    train_data = merge(question_path=train_question_file, answers_path=train_answer_file, image_ids_path=train_images_file, ppl_dir=ppl_path, resnet_dir=resnet_dir)
    
    train_annotation_save_path = "/home/shiina/data/nips/cocoqa/annotation/raw"
    os.makedirs(train_annotation_save_path, exist_ok=True)

    with open(os.path.join(train_annotation_save_path, "train_instances.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    print("---------train split--------")
    print("Statistics: all:{}".format(len(train_data)))

    val_answer_file = "/home/shiina/data/cocoqa/test/answers.txt"
    val_question_file = "/home/shiina/data/cocoqa/test/questions.txt"
    val_images_file = "/home/shiina/data/cocoqa/test/img_ids.txt"
    ppl_path = "/home/shiina/data/nips/vqa2/features/proposals/val"
    resnet_dir = "/home/shiina/data/iclr_further/img_resnet_feats/val"
    val_data = merge(question_path=val_question_file, answers_path=val_answer_file, image_ids_path=val_images_file, ppl_dir=ppl_path, resnet_dir=resnet_dir)
    val_annotation_save_path = "/home/shiina/data/nips/cocoqa/annotation/raw"

    with open(os.path.join(val_annotation_save_path, "val_instances.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    print("------val split--------")
    print("Statistics: all:{}".format(len(val_data)))
    exit(0)




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