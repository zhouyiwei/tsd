#!/usr/bin/env python
import numpy as np
SEED = 1013
np.random.seed(SEED)
from parameters import *
import os
from utils import *
from nltk.tokenize import StanfordTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dense, Input, Flatten, Dropout, Merge
from nltk.tokenize import TweetTokenizer
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from model import *
import json
from collections import Counter
import copy

tokenizer = TweetTokenizer()

train_data_file = 'data/semeval2016-task6-training_trial_data.txt'
test_data_file = 'data/SemEval2016-Task6-subtaskA-testdata-gold.txt'
TARGETS = ['Hillary Clinton', 'Legalization of Abortion', 'Atheism', 'Feminist Movement', 'Climate Change is a Real Concern']
# TARGETS = ['']
VOTE_MODE = False
EARLY_STOPPING = False

def train_and_test(target, config):
    # print "loading the data..."
    sentence_maxlen = 0
    target_maxlen = 0
    x_s_tokens = []
    x_t_tokens = []
    y = []
    with open(train_data_file, 'r') as inputfile:
        for each_line in inputfile:
            each_line_ = each_line.decode('latin-1').replace('#SemST', '').strip()
            each_line_ = each_line_.split('\t')
            if each_line_[0].strip() != 'ID' and target in each_line_[1].strip():
                s_tokens = tokenizer.tokenize(preprocess_tweets(each_line_[2].strip()))
                if len(s_tokens) > sentence_maxlen:
                    sentence_maxlen = len(s_tokens)
                x_s_tokens.append(s_tokens)
                t_tokens = each_line_[1].strip().lower().split()
                if len(t_tokens) > target_maxlen:
                    target_maxlen = len(t_tokens)
                x_t_tokens.append(t_tokens)
                y.append(classes[each_line_[3].strip()])
    x_s_test_tokens = []
    x_t_test_tokens = []
    y_test = []
    with open(test_data_file, 'r') as inputfile:
        for each_line in inputfile:
            each_line_ = each_line.decode('latin-1').replace('#SemST', '').strip()
            each_line_ = each_line_.split('\t')
            if each_line_[0] != 'ID' and target in each_line_[1].strip():
                s_test_tokens = tokenizer.tokenize(preprocess_tweets(each_line_[2].strip()))
                if len(s_test_tokens) > sentence_maxlen:
                    sentence_maxlen = len(s_test_tokens)
                x_s_test_tokens.append(s_test_tokens)
                t_val_tokens = each_line_[1].strip().lower().split()
                if len(t_val_tokens) > target_maxlen:
                    target_maxlen = len(t_val_tokens)
                x_t_test_tokens.append(t_val_tokens)
                y_test.append(classes[each_line_[3].strip()])

    # print "preparing data and loading word embedding..."
    tokens2index, index2tokens = build_vocabulary(x_s_tokens+x_t_tokens+x_s_test_tokens+x_t_test_tokens)
    x_s = [[tokens2index[each_t] if each_t in tokens2index else 0 for each_t in each_s] for each_s in x_s_tokens]
    x_s = pad_sequences(x_s, maxlen=sentence_maxlen)
    x_s_test = [[tokens2index[each_t] if each_t in tokens2index else 0 for each_t in each_s] for each_s in x_s_test_tokens]
    x_s_test = pad_sequences(x_s_test, maxlen=sentence_maxlen)
    x_t = [[tokens2index[each_t] if each_t in tokens2index else 0 for each_t in each_s] for each_s in x_t_tokens]
    x_t = pad_sequences(x_t, maxlen=target_maxlen)
    x_t_test = [[tokens2index[each_t] if each_t in tokens2index else 0 for each_t in each_s] for each_s in x_t_test_tokens]
    x_t_test = pad_sequences(x_t_test, maxlen=target_maxlen)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_s = x_s[shuffle_indices]
    x_t = x_t[shuffle_indices]
    y = np.asarray(y)
    y = y[shuffle_indices]
    y_test = np.asarray(y_test)

    f1_val_scores = []
    y_preds_l = []
    y_preds_r = []

    config['sentence_maxlen'] = sentence_maxlen
    config['target_maxlen'] = target_maxlen
    config['stances_num'] = len(y[0])
    config['classes'] = classes_

    kfold = StratifiedKFold(n_splits=5)
    for train_idx, val_idx in kfold.split(x_s, classes_[y.argmax(1)]):
        x_s_train = x_s[train_idx]
        x_t_train = x_t[train_idx]
        y_train = y[train_idx]
        x_s_val = x_s[val_idx]
        x_t_val = x_t[val_idx]
        y_val = y[val_idx]

        results = {}
        model = GatedBiGRUCNN(config)
        if config['embedding'] == 't':
            embedding_matrix_twitter = np.zeros((len(index2tokens.keys())+1, config['embedding_dim']))
            word_embeddings_twitter = load_glove_embeddings(embedding='twitter', dim=config['embedding_dim'])
            for each_key in index2tokens.keys():
                if index2tokens[each_key] in word_embeddings_twitter:
                    embedding_matrix_twitter[each_key] = word_embeddings_twitter[index2tokens[each_key]]
            model.compile(embedding_matrix=embedding_matrix_twitter)
        elif config['embedding'] == 'w':
            embedding_matrix_wikipedia = np.zeros((len(index2tokens.keys())+1, config['embedding_dim']))
            word_embeddings_wikipedia = load_glove_embeddings(embedding='wikipedia', dim=config['embedding_dim'])
            for each_key in index2tokens.keys():
                if index2tokens[each_key] in word_embeddings_wikipedia:
                    embedding_matrix_wikipedia[each_key] = word_embeddings_wikipedia[index2tokens[each_key]]
            model.compile(embedding_matrix=embedding_matrix_wikipedia)

        if not EARLY_STOPPING:
            for i in range(1, config['epochs']+1):
                model.fit(x_s_train, x_t_train, y_train, x_s_val, x_t_val, y_val, epoch=1)
                score_val, y_val_l, y_val_pred_l, y_val_pred_r = model.evaluate(x_s_val, x_t_val, y_val)
                score_test, y_test_l, y_test_pred_l, y_test_pred_r = model.evaluate(x_s_test, x_t_test, y_test)
                results[i] = {'y_val': y_val_l.tolist(), 'y_val_pred': y_val_pred_l.tolist(), 'y_test': y_test_l.tolist(), 'y_test_pred': y_test_pred_l.tolist(), 'score_val': score_val, 'score_test': score_test, 'y_test_pred_real': y_test_pred_r}
        else:
            i = 0
            model.fit(x_s_train, x_t_train, y_train, x_s_val, x_t_val, y_val, epoch=config['epochs'])
            score_val, y_val_l, y_val_pred_l, y_val_pred_r = model.evaluate(x_s_val, x_t_val, y_val)
            score_test, y_test_l, y_test_pred_l, y_test_pred_r = model.evaluate(x_s_test, x_t_test, y_test)
            results[i] = {'y_val': y_val_l.tolist(), 'y_val_pred': y_val_pred_l.tolist(), 'y_test': y_test_l.tolist(), 'y_test_pred': y_test_pred_l.tolist(), 'score_val': score_val, 'score_test': score_test, 'y_test_pred_real': y_test_pred_r}

        del model
        cur_highest_score_val = -1
        cur_y_test_pred = []
        cur_y_test_pred_real = []
        for i in results.keys():
            if results[i]['score_val'] > cur_highest_score_val:
                cur_highest_score_val = results[i]['score_val']
                cur_y_test_pred = results[i]['y_test_pred']
                cur_y_test_pred_real = results[i]['y_test_pred_real']
        f1_val_scores.append(cur_highest_score_val)
        y_preds_l.append(cur_y_test_pred)
        y_preds_r.append(cur_y_test_pred_real)

    y_preds_l = np.asarray(y_preds_l)
    y_preds_l = np.swapaxes(y_preds_l, 0, 1)
    y_preds_r = np.asarray(y_preds_r)
    y_preds_r = np.sum(y_preds_r, axis=0)
    y_pred = []
    if VOTE_MODE:
        for each_pred in y_preds_l:
            y_pred.append(Counter(each_pred).most_common(1)[0][0])
    else:
        y_pred = classes_[y_preds_r.argmax(1)].tolist()
    return np.mean(f1_val_scores), f1_score(classes_[y_test.argmax(1)].tolist(), y_pred, average='macro', labels=['FAVOR', 'AGAINST']), classes_[y_test.argmax(1)].tolist(), y_pred


if __name__ == '__main__':
    val_scores = []
    test_scores = []
    y_test = []
    y_pred = []
    for each_target in TARGETS:
        cur_highest_score_val = -1
        cur_test_score = None
        cur_pred = None
        cur_test = None
        cur_setting = None
        config = {}
        dims = [256]  # 64
        dropouts = [0.5]
        batch_sizes = [16] # 16
        hidden_units = [0]
        config['embedding'] = 'w'
        config['embedding_dim'] = 300  # 100
        config['epochs'] = 50
        config['hidden_units_num'] = 0
        for each_dim in dims:
            for each_dropout in dropouts:
                for each_batch in batch_sizes:
                    config['dim'] = each_dim
                    config['dropout'] = each_dropout
                    config['batch_size'] = each_batch
                    print '++++++++++++++++++++'
                    print 'Results for '+each_target+':'
                    print config['embedding'], config['embedding_dim'], config['batch_size'], config['dim'], config['dropout']
                    a, b, c, d = train_and_test(each_target, config)
                    print '+++++'
                    print a
                    print b
                    print '+++++'
                    print '++++++++++++++++++++'
                    if a > cur_highest_score_val:
                        cur_highest_score_val = a
                        cur_test_score = b
                        cur_test = c
                        cur_pred = d
                        cur_setting = copy.deepcopy(config)
        val_scores.append(cur_highest_score_val)
        test_scores.append(cur_test_score)
        y_test.extend(cur_test)
        y_pred.extend(cur_pred)
        print "============================"
        print 'Best results for '+each_target+':'
        print cur_setting['embedding'], cur_setting['embedding_dim'], cur_setting['batch_size'], cur_setting['dim'], cur_setting['dropout']
        print cur_highest_score_val
        print cur_test_score
        print "============================"
    print "============================"
    print 'Final results:'
    print np.mean(val_scores)
    print np.mean(test_scores)
    print f1_score(y_test, y_pred, average='macro', labels=['FAVOR', 'AGAINST'])
    print "============================"
    if TARGETS == ['']:
        pred = y_pred[220: 389]
        test = y_test[220: 389]
        print f1_score(test, pred, average='macro', labels=['FAVOR', 'AGAINST'])


