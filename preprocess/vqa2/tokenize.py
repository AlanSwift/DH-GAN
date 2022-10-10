### Note: this code is duplicated. Please refer to annotator.ipynb for more details


import re, json, pickle, copy

def stanfordcorenlp_pos_tag(sentence: str, nlp_processor):
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\?', ' ', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    props = {
        'annotators': 'ssplit,tokenize,pos,lemma',
        "tokenize.options":
            "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
        "tokenize.whitespace": False,
        'ssplit.isOneSentence': True,
        'outputFormat': 'json'
    }
    ret = nlp_processor.annotate(sentence.lower(), props)
    try:
        pos_dict = json.loads(ret)['sentences'][0]['tokens']

    except:
        return []

    ret = []
    for token in pos_dict:
        ret.append({
            "word_idx": token['index'] - 1,
            "word": token["word"],
            "lemma": token["lemma"],
            "pos": token["pos"]
        })
    return ret


if __name__ == "__main__":
    from stanfordcorenlp import StanfordCoreNLP
    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)
    print(nlp_parser)
    raw_file = "/home/shiina/data/nips/vqa2/annotation/raw/train_instances.pkl"
    save_file = "/home/shiina/data/nips/vqa2/annotation/raw/train_instances_tokenized.pkl"
    with open(raw_file, "rb") as f:
        content = pickle.load(f)
    ret = {}
    for qid, inst in content.items():
        question = inst["question"].lower()
        answer = inst["multiple_choice_answer"].lower()
        img_id = inst["image_id"]

        parsed_results = stanfordcorenlp_pos_tag(question, nlp_parser)
        question_tokens = [token_inst["word"] for token_inst in parsed_results]
        question_lemmas = [token_inst["lemma"] for token_inst in parsed_results]
        parsed_results = stanfordcorenlp_pos_tag(answer, nlp_parser)
        answer_tokens = [token_inst["word"] for token_inst in parsed_results]
        answer_lemmas = [token_inst["lemma"] for token_inst in parsed_results]
        inst["question_tokens"] = question_tokens
        inst["question_lemmas"] = question_lemmas
        inst["answer_tokens"] = answer_tokens
        inst["answer_lemmas"] = answer_lemmas
        inst["visual_hint"] = [0 for _ in range(36)] # fake visual hints, due to no use in inference stage
        ret[qid] = copy.deepcopy(inst)
    with open(save_file, "wb") as f:
        pickle.dump(ret, f)
    
    pass
