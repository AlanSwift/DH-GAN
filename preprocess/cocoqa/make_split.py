"""
    Meta-information:
        train_split_path: "/home/shiina/data/nips/vqa2/processed/train_split.pkl"
        val_split_path: "/home/shiina/data/nips/vqa2/processed/val_split.pkl"
        test_split_path: "/home/shiina/data/nips/vqa2/processed/test_split.pkl"
"""


import os
import pickle
import random

MAX_PPL = 36

def filter_duplicate(raw_split):
    insts_collect = {}
    for qid, inst in raw_split.items():
        image_id = inst["image_id"]
        answer = inst["multiple_choice_answer"].lower().strip()
        if answer == "":
            continue
        unique_key = (image_id, answer)
        if unique_key not in insts_collect:
            insts_collect[unique_key] = [inst]
        else:
            insts_collect[unique_key].append(inst)
    ret = []
    for unique_key, inst_list in insts_collect.items():
        select = random.choice(inst_list)
        ret.append(select)
    return ret

def remove_invalid(split):
    ret = []
    global MAX_PPL
    for inst in split:
        with open(inst['ppl_path'], "rb") as f:
            ppl_json = pickle.load(f)

        ppl_spatial_info = ppl_json["spatial_features"]

        ppl_features = ppl_json["features"]
        if ppl_spatial_info.shape[0] != MAX_PPL:
            continue
        if ppl_features.shape[0] != MAX_PPL:
            continue
        if ppl_features.shape[0] == 35:
            print(ppl_features.shape, "+++++")
        ret.append(inst)
    return ret   
    pass

if __name__ == "__main__":
    random.seed(1234)
    train_raw_split_path = "/home/shiina/data/nips/vqa2/annotation/cache/train_instances_with_doublehints.pkl"
    train_split_save_path = "/home/shiina/data/nips/vqa2/processed/train_split.pkl"
    val_raw_split_path = "/home/shiina/data/nips/vqa2/annotation/cache/val_instances_with_doublehints.pkl"
    val_split_save_path = "/home/shiina/data/nips/vqa2/processed/val_split.pkl"
    test_split_save_path = "/home/shiina/data/nips/vqa2/processed/test_split.pkl"
    val_test_ratio = 0.1
    with open(train_raw_split_path, "rb") as f:
        train_raw_split = pickle.load(f)
    train_split = filter_duplicate(raw_split=train_raw_split)
    train_split = remove_invalid(train_split)

    print("---- After filtering duplicate items, the Train split's statistics -----")
    print("Original number: {}, Saved number: {}".format(len(train_raw_split), len(train_split)))

    with open(train_split_save_path, "wb") as f:
        pickle.dump(train_split, f)


    with open(val_raw_split_path, "rb") as f:
        val_raw_split = pickle.load(f)
    val_split = filter_duplicate(val_raw_split)

    random.shuffle(val_split)
    val_num = int(len(val_split)*val_test_ratio)
    validate_split = val_split[:val_num]
    test_split = val_split[val_num:]
    validate_split = remove_invalid(validate_split)
    test_split = remove_invalid(test_split)
    with open(val_split_save_path, "wb") as f:
        pickle.dump(validate_split, f)
    with open(test_split_save_path, "wb") as f:
        pickle.dump(test_split, f)

    print("---- After filtering duplicate items, the Val split's statistics -----")
    print("Original number: {}, Saved number: {}".format(len(val_raw_split), len(val_split)))

    print("+++++ Now split the Val split into Validate and Test splits according to ratio: {} +++++".format(val_test_ratio))
    print("Validate number: {}, Test number: {}".format(len(validate_split), len(test_split)))
    pass