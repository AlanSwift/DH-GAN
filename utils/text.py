import re
import nltk
import json


def tokenize(sentence: str) -> list:
    """Tokenizes a sentence into words.

    Args:
        sentence: A string of words.

    Returns:
        A list of words.
    """
    if len(sentence) == 0:
        return []
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\?', ' ', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    tokens = nltk.tokenize.word_tokenize(sentence.strip().lower())
    return tokens


def stanfordcorenlp_pos_tag(sentence: str, nlp_processor):
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\?', ' ', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    props = {
        'annotators': 'ssplit,tokenize,pos,lemma',
        "tokenize.options":
            "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
        "tokenize.whitespace": False,
        'ssplit.isOneSentence': True,
        'outputFormat': 'json'
    }
    ret = nlp_processor.annotate(sentence, props)
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
