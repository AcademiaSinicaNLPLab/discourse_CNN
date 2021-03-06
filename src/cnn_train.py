#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import cPickle
import numpy as np
np.random.seed(10)
import pandas
from word2vec.word2vec import Word2Vec
from sklearn.cross_validation import StratifiedKFold, train_test_split

from classifier.cnn import Kim_CNN
from classifier.cnn import ConjWeight_CNN
from classifier.cnn import ConjWeightOneVec_CNN
from classifier.cnn import ConjWeightNegVec_CNN
from classifier.cnn import ConjWeightAllVec_CNN
from classifier.cnn import ConjWeightTwoVec_CNN
from classifier.cnn import ConjWeight_CNN2
from classifier.cnn import RNN

from feature.extractor import CNNExtractor
from feature.extractor import ConjWeightCNNExtractor
from feature.extractor import ConjWeightOneVecCNNExtractor
from feature.extractor import ConjWeightNegVecCNNExtractor
from feature.extractor import ConjWeightAllVecCNNExtractor
from feature.extractor import ConjWeightTwoVecCNNExtractor
from feature.extractor import ConjWeightCNNExtractor2
from feature.extractor import ConjWeightAllVecCNNExtractor2
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback

import logging
logging.basicConfig(level=logging.DEBUG)


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
CONJCORPUS_DIR = os.path.join(MODULE_DIR, '..', 'conjcorpus')
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

class RecordTest(Callback):
    def __init__(self, X_test, y_test):
        super(RecordTest, self).__init__()
        self.X_test, self.y_test = X_test, y_test
        self.best_val_acc =0.
        self.test_acc = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_acc')
        if current is None:
            warnings.warn('val_acc required')

        pred = self.model.predict(self.X_test, verbose=0)
        test_acc = np.mean(np.equal(np.argmax(pred, axis=-1), np.argmax(self.y_test, axis=-1)))
        if current > self.best_val_acc:
            self.best_val_acc = current
            self.test_acc = test_acc
        print
        print "Test acc(this epoch)/Best test acc: {}/{}".format(test_acc, self.test_acc)

def separateX(X, dtype='int32'):
    if len(X.shape)==2 and X.dtype==np.dtype('int'):
        return X

    res = []

    for channel in range(X.shape[1]):
        row = len(X[:,channel])
        res.append(np.concatenate(X[:,channel]).reshape(row,-1).astype(dtype))
    return res

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    np.random.seed(args.seed)

    dataset = args.dataset
    if os.path.isfile(os.path.join(CORPUS_DIR, dataset+'.pkl')):
        corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    else:
        corpus = pandas.read_pickle(os.path.join(CONJCORPUS_DIR, dataset+'.pkl'))
    sentences, labels = list(corpus.sentence), list(corpus.label)

    if len(set(corpus.split.values))==1:
        split = None
    else:
        split = corpus.split.values

    m=4
    if m==1:
        cnn_extractor = CNNExtractor()
    elif m==2:
        cnn_extractor = ConjWeightCNNExtractor()
    elif m==3:
        cnn_extractor = ConjWeightNegVecCNNExtractor()
    elif m==4:
        cnn_extractor = ConjWeightAllVecCNNExtractor()
    elif m==5:
        cnn_extractor = ConjWeightOneVecCNNExtractor()
    elif m==6:
        cnn_extractor = ConjWeightTwoVecCNNExtractor()
    elif m==7:
        cnn_extractor = ConjWeightCNNExtractor2()
    elif m==9:
        cnn_extractor = CNNExtractor()

    X, y = cnn_extractor.extract_train(sentences, labels)
    # sys.exit(-1)

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

    common_args = dict(vocabulary_size=cnn_extractor.vocabulary_size,
                       maxlen=maxlen,
                       filter_length=[3,4,5],
                       embedding_dim=embedding_dim,
                       nb_class=len(cnn_extractor.literal_labels),
                       drop_out_prob=0.5,
                       nb_filter=100,
                       maxnorm=9,
                       embedding_weights=W)

    if m==1:
        clf = Kim_CNN(**common_args)
    elif m==2:
        clf = ConjWeight_CNN(**common_args)
    elif m==3:
        clf = ConjWeightNegVec_CNN(**common_args)
    elif m==4:
        clf = ConjWeightAllVec_CNN(weight_range=(cnn_extractor.min_weight, cnn_extractor.max_weight), **common_args)
    elif m==5:
        clf = ConjWeightOneVec_CNN(**common_args)
    elif m==6:
        clf = ConjWeightTwoVec_CNN(weight_range=(cnn_extractor.min_weight, cnn_extractor.max_weight), **common_args)
    elif m==7:
        clf = ConjWeight_CNN2(weight_range=(cnn_extractor.min_weight, cnn_extractor.max_weight), **common_args)
    elif m==9:
        clf = RNN(**common_args)

    print 'm:{}, Extractor: {}, Clf: {}'.format(m, cnn_extractor.__class__.__name__, clf.__class__.__name__)

    # if split is None:
    if True:    # always use CV
        test_acc = []
        cv = StratifiedKFold(y, n_folds=args.fold, shuffle=True)
        # cv = StratifiedKFold(y, n_folds=10, shuffle=True)
        y = to_categorical(y)
        for train_ind, test_ind in cv:
            X_train, X_test = X[train_ind], X[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]
            X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1)

            X_train = separateX(X_train)
            X_dev = separateX(X_dev)
            X_test = separateX(X_test)

            callback = RecordTest(X_test, y_test)
            clf.fit(X_train, y_train,
                    batch_size=50,
                    nb_epoch=args.epoch,
                    validation_data=(X_dev, y_dev),
                    callbacks=[callback], verbose=1)

            test_acc.append(callback.test_acc)
            print test_acc, np.average(test_acc)
    else:
        y = to_categorical(y)
        train_ind, dev_ind, test_ind = (split=='train', split=='dev', split=='test')
        X_train, X_dev, X_test = X[train_ind], X[dev_ind], X[test_ind]
        y_train, y_dev, y_test = y[train_ind], y[dev_ind], y[test_ind]
        print "Train/Dev/Test:{}/{}/{}".format(len(X_train), len(X_dev), len(X_test))

        X_train = separateX(X_train)
        X_dev = separateX(X_dev)
        X_test = separateX(X_test)

        callback = RecordTest(X_test, y_test)
        clf.fit(X_train, y_train,
                batch_size=50,
                nb_epoch=args.epoch,
                validation_data=(X_dev, y_dev),
                callbacks=[callback], verbose=1)
