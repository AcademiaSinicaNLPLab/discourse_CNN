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

from feature.extractor import CNNExtractor
from feature.extractor import ConjWeightCNNExtractor
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

class RecordTest2(Callback):
    def __init__(self, sentences, markers, X_test, y_test):
        super(RecordTest2, self).__init__()
        self.X_test, self.y_test = X_test, y_test
        self.best_val_acc =0.
        self.test_acc = 0

        self.markers = markers
        self.marker_count = {marker: 0. for marker in markers}
        self.marker_total = {marker: 0. for marker in markers}

        self.marker_index = {marker:np.array([False]*len(sentences)) for marker in markers}
        # for i, sentence in enumerate(sentences):
        #     for marker in markers:
        #         if marker in sentence:
        #             self.marker_index[marker][i] = True
        marker_set = set(markers)
        for i, sentence in enumerate(sentences):
            wordarray = set(sentence.split())
            for marker in marker_set:
                insect = wordarray.intersection(marker_set)
                if len(insect)==1:
                    self.marker_index[insect.pop()][i] = True
        for marker in markers:
            self.marker_total[marker] = float(np.sum(self.marker_index[marker]))

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_acc')
        if current is None:
            warnings.warn('val_acc required')

        pred = self.model.predict(self.X_test, verbose=0)
        equal = np.equal(np.argmax(pred, axis=-1), np.argmax(self.y_test, axis=-1))

        test_acc = np.mean(equal)
        if current > self.best_val_acc:
            self.best_val_acc = current
            self.test_acc = test_acc
            for marker in self.markers:
                self.marker_count[marker] = np.sum(equal[self.marker_index[marker]])

        print
        print "Test acc(this epoch)/Best test acc: {}/{}".format(test_acc, self.test_acc)

class Pattern(object):
    def __init__(self, dim):
        self.count = 0 
        self.index = np.array([False]*dim)

    @property
    def total(self):
        return np.sum(self.index)

class RecordTest(Callback):
    def __init__(self, sentences, X_test, y_test):
        super(RecordTest, self).__init__()
        self.X_test, self.y_test = X_test, y_test
        self.best_val_acc =0.
        self.test_acc = 0

        conj_fol_set = set(['but', 'however', 'nevertheless', 'otherwise', 'yet', 'still', 'nonetheless', 'therefore', 'furthermore', 'consequently', 'thus', 'subsequently', 'eventually', 'hence'])

        all_set = set(['but', 'however', 'nevertheless', 'otherwise', 'yet', 'still', 'nonetheless', 'therefore', 'furthermore', 'consequently', 'thus', 'subsequently', 'eventually', 'hence', 'till', 'until', 'despite', 'though', 'although', 'unless'])

        neg_set = set(['n\'t', 'not', 'neither', 'never', 'no', 'nor'])

        self.only_conj = Pattern(len(sentences))
        self.conj_neg = Pattern(len(sentences))
        self.only_neg = Pattern(len(sentences))

        for i, sentence in enumerate(sentences):
            wordarray = sentence.split()
            if len(conj_fol_set.intersection(wordarray))>0:

                if len(neg_set.intersection(wordarray))>0:
                    for j in range(len(wordarray)):
                        if wordarray[j] in conj_fol_set:
                            break
                    for k in range(len(wordarray)):
                        if wordarray[k] in neg_set:
                            break
                    if j<k:
                        self.conj_neg.index[i] = True
                else:
                    self.only_conj.index[i] = True
            elif len(all_set.intersection(wordarray))==0:
                if len(neg_set.intersection(wordarray))>0:
                    self.only_neg.index[i] = True

        print sentences[self.conj_neg.index]
        sys.exit(-1)

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_acc')
        if current is None:
            warnings.warn('val_acc required')

        pred = self.model.predict(self.X_test, verbose=0)
        equal = np.equal(np.argmax(pred, axis=-1), np.argmax(self.y_test, axis=-1))

        test_acc = np.mean(equal)
        if current > self.best_val_acc:
            self.best_val_acc = current
            self.test_acc = test_acc
            self.only_conj.count = np.sum(equal[self.only_conj.index])
            self.conj_neg.count = np.sum(equal[self.conj_neg.index])
            self.only_neg.count = np.sum(equal[self.only_neg.index])

        # print
        # print "Test acc(this epoch)/Best test acc: {}/{}".format(test_acc, self.test_acc)

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

    m=2
    if m==1:
        cnn_extractor = CNNExtractor()
    elif m==2:
        cnn_extractor = ConjWeightCNNExtractor()

    X, y = cnn_extractor.extract_train(sentences, labels)
    # sys.exit(-1)

    if args.debug:
        W = None
        embedding_dim = 300
    else:
        logging.debug('loading embedding..')
        W = load_embedding(cnn_extractor.vocabulary,
                           cache_file_name=os.path.join(CACHE_DIR, dataset + '_emb.pkl'))
        embedding_dim=W.shape[1]

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

    print 'm:{}, Extractor: {}, Clf: {}'.format(m, cnn_extractor.__class__.__name__, clf.__class__.__name__)

    test_acc = []

    only_conj = []
    conj_neg = []
    only_neg = []
    
    cv = StratifiedKFold(y, n_folds=args.fold, shuffle=True)
    y = to_categorical(y)
    for train_ind, test_ind in cv:
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1)

        X_train = separateX(X_train)
        X_dev = separateX(X_dev)
        X_test = separateX(X_test)

        callback = RecordTest(np.asarray(sentences)[test_ind], X_test, y_test)
        clf.fit(X_train, y_train,
                batch_size=50,
                nb_epoch=args.epoch,
                validation_data=(X_dev, y_dev),
                callbacks=[callback], verbose=1)

        only_conj.append(callback.only_conj)
        conj_neg.append(callback.conj_neg)
        only_neg.append(callback.only_neg)

        test_acc.append(callback.test_acc)
        print test_acc, np.average(test_acc)

    def pattern_summary(pattern_list): 
        count = sum([pattern.count for pattern in pattern_list])
        total = float(sum(pattern.total for pattern in pattern_list))
        print "{}/{}/{}".format(count, total, count/total)

    print 'only_conj',
    pattern_summary(only_conj)
    print 'conj_neg',
    pattern_summary(conj_neg)
    print 'only_neg',
    pattern_summary(only_neg)
