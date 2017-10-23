import re
import numpy as np
from nltk.tokenize import TweetTokenizer

def preprocess_tweets(text):
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"[8:=;]['`\-]?[)d]+|[)d]+['`\-]?[8:=;]", "<SMILE>", text)
    text = re.sub(r"[8:=;]['`\-]?p+", "<LOLFACE>", text)
    text = re.sub(r"[8:=;]['`\-]?\(+|\)+['`\-]?[8:=;]", "<SADFACE>", text)
    text = re.sub(r"[8:=;]['`\-]?[\/|l*]", "<NEUTRALFACE>", text)
    text = re.sub(r"<3","<HEART>", text)
    text = re.sub(r"/"," / ", text)
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    p = re.compile(r"#\S+")
    text = p.sub(lambda s: "<HASHTAG> "+s.group()+" <ALLCAPS>" if s.group()[1:].isupper() else " ".join(["<HASHTAG>"]+re.split(r"([A-Z][^A-Z]*)", s.group()[1:])), text)
    text = re.sub(r"([!?.]){2,}", r"\1 <REPEAT>", text)
    text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <ELONG>", text)
    return text.lower()


def load_glove_embeddings(embedding='twitter', dim=50):
    if embedding == 'twitter':
        file_path = 'tools/glove.twitter.27B.'+str(dim)+'d.txt'
    else:
        file_path = 'tools/glove.6B.'+str(dim)+'d.txt'
    word_embeddings = {}
    with open(file_path, 'r') as inputfile:
        for each_line in inputfile:
            each_line_ = each_line.split(' ')
            word_embeddings[each_line_[0].decode('utf8')] = np.asarray(each_line_[1:], dtype='float32')
    return word_embeddings


def build_vocabulary(tokens_list):
    all_tokens = set()
    for each_s in tokens_list:
        all_tokens |= set(each_s)
    all_tokens = list(all_tokens)
    tokens2index = {}
    index2tokens = {}
    for i in range(len(all_tokens)):
        tokens2index[all_tokens[i]] = i+1  # start from 1, 0 reserved for padding sequences
        index2tokens[i+1] = all_tokens[i]
    return tokens2index, index2tokens

def extract_useful_tweets():
    ids_tweets = {}
    with open('semeval2016-task6/downloaded_Donald_Trump_all.txt', 'r') as inputfile:
        for each_line in inputfile:
            each_line_ = each_line.decode('latin-1').split('\t')
            ids_tweets.setdefault(each_line_[0].strip(), each_line_[1].strip())
            if len(each_line_[1].strip()) > ids_tweets[each_line_[0].strip()]:
                ids_tweets[each_line_[0].strip()] = each_line_[1].strip()
    with open('semeval2016-task6/Donald_Trump.txt', 'r') as inputfile:
        with open('semeval2016-task6/downloaded_Donard_Trump.txt', 'w') as outputfile:
            for each_line in inputfile:
                outputfile.write((each_line.strip()+'\t'+ids_tweets[each_line.strip()]+'\n').encode('latin-1'))


if __name__ == '__main__':
    # print TweetTokenizer().tokenize(preprocess_tweets('I TEST all kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!'))
    extract_useful_tweets()