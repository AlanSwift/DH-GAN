import pickle

if __name__ == "__main__":
    train_split_path = "/home/shiina/shiina/question/ban-vqa/data/nips/10/cache/train_split.pkl"
    save_path = "/home/shiina/shiina/question/ban-vqa/data/nips/10/cache/answer_list.pkl"
    with open(train_split_path, "rb") as f:
        split = pickle.load(f)
    
    cnt = 0
    answer_dict = {"UNK": cnt}
    for inst in split:
        ans = inst["answer"]
        if ans not in answer_dict:
            answer_dict[ans] = cnt
            cnt += 1

    print("Answer list", len(answer_dict.keys()))
    with open(save_path, "wb") as f:
        pickle.dump(answer_dict, f)

    pass