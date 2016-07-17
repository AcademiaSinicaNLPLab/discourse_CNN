#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import cPickle
import numpy as np
import pandas
from word2vec.word2vec import Word2Vec
from sklearn.grid_search import ParameterGrid
from joblib import Parallel, delayed

from classifier.cnn import Kim_CNN
from classifier.cnn import RNN

from feature.extractor import CNNExtractor
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback

import logging
logging.basicConfig(level=logging.DEBUG)


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
CACHE_DIR = os.path.join(MODULE_DIR, '..', 'cache')

def load_embedding(vocabulary, cache_file_name):
    if os.path.isfile(cache_file_name):
        with open(cache_file_name) as f:
            return cPickle.load(f)
    else:
        res = np.random.uniform(low=-0.05, high=0.05, size=(len(vocabulary), 300))
        w2v = Word2Vec()
        for word in vocabulary.keys():
            if word in w2v:
                ind = vocabulary[word]
                res[ind] = w2v[word]
        with open(cache_file_name, 'w') as f:
            cPickle.dump(res, f)
        return res

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', help='dataset name')
    parser.add_argument('-e', '--epoch', type=int, default=20, help='number of epoch')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-d', '--debug', action='store_true', help='fast debug mode')
    parser.add_argument('-cv', '--fold', type=int, default=10, help='K fold')
    return parser.parse_args(argv[1:])

from keras.callbacks import Callback
class CheckAcc(Callback):
    def __init__(self, maxepoch=12, minacc=0.96):
        self.maxepoch=maxepoch
        self.minacc=minacc

    def on_epoch_end(self, epoch, logs={}):
        acc = logs.get('acc')
        if acc>self.minacc or epoch>=self.maxepoch:
            print epoch, acc
            self.model.stop_training = True

def fit_wrapper(clfcls, X_train, y_train, X_dev, y_dev, batch_size=50, nb_epoch=50, verbose=1, maxepoch=30, minacc=0.97, **non_trainable_args):
    def fit(**trainable_args):
        args = non_trainable_args.copy()
        args.update(trainable_args)
        clf = clfcls(callbacks=[CheckAcc(maxepoch, minacc)], **args)
        clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose) 
        score = clf.score(X_dev, y_dev, verbose=0)
        print trainable_args, score
        print '-'*20
        return score

    return fit

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    np.random.seed(args.seed)

    dataset = args.dataset
    if os.path.isfile(os.path.join(CORPUS_DIR, dataset+'.pkl')):
        corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    sentences, labels = list(corpus.sentence), list(corpus.label)

    if len(set(corpus.split.values))==1:
        split = None
    else:
        split = corpus.split.values

    cnn_extractor = CNNExtractor()
    X, y = cnn_extractor.extract_train(sentences, labels)

    if args.debug:
        logging.debug('Embedding is None!!!!!!!!!!!!')
        W = None
        embedding_dim = 300
    else:
        logging.debug('loading embedding..')
        W = load_embedding(cnn_extractor.vocabulary,
                           cache_file_name=os.path.join(CACHE_DIR, dataset + '_emb.pkl'))
        embedding_dim=W.shape[1]

    logging.debug('embedding loaded..')

    maxlen = cnn_extractor.maxlen

    y = to_categorical(y)
    train_ind, dev_ind = (split=='train', split=='dev')
    X_train, X_dev = X[train_ind], X[dev_ind]
    y_train, y_dev = y[train_ind], y[dev_ind]

    # fit = fit_wrapper(Kim_CNN, X_train, y_train, X_dev, y_dev, batch_size=50, nb_epoch=100, verbose=0,
    fit = fit_wrapper(RNN, X_train, y_train, X_dev, y_dev, batch_size=50, nb_epoch=100, verbose=0,
                      vocabulary_size=cnn_extractor.vocabulary_size,
                      maxlen=maxlen,
                      embedding_dim=embedding_dim,
                      nb_class=len(cnn_extractor.literal_labels),
                      embedding_weights=W,
                      maxepoch=15,
                      minacc=0.97)

    params = list(ParameterGrid(dict(filter_length=[[2,3,4]],
                                     drop_out_prob=[0, 0.2, 0.5, 0.7],
                                     nb_filter=[50,100,150],
                                     maxnorm=[9])))

    # params = list(ParameterGrid(dict(filter_length=[[2,3,4], [4,5,6], [6,7,8]],
    #                                  drop_out_prob=[0, 0.2, 0.5, 0.7],
    #                                  nb_filter=[50,100,150],
    #                                  maxnorm=[9])))

    res = Parallel(n_jobs=1)(delayed(fit)(**param) for param in params)
    res = sorted(zip(params, res), key=lambda x: x[1], reverse=True)

    for r in res:
        print r
