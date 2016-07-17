#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import numpy as np
from sklearn.svm import LinearSVC
import pandas
from feature.extractor import feature_fuse, W2VExtractor
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
import logging
from joblib import Parallel, delayed
from sklearn.grid_search import ParameterGrid
logging.basicConfig(level=logging.DEBUG)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
MODEL_DIR = os.path.join(MODULE_DIR, '..', 'model')

CLF_CLASS = LinearSVC

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', help='dataset name')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-cv', '--fold', type=int, default=10, help='K fold')
    return parser.parse_args(argv[1:])


def fit_wrapper(X_train, y_train, X_dev, y_dev, X_test, y_test):
    def fit(**args):
        clf = CLF_CLASS(**args)
        clf.fit(X_train, y_train) 
        dev_score = clf.score(X_dev, y_dev)
        test_score = clf.score(X_test, y_test)
        print args, dev_score, test_score
        print '-'*20
        return args, dev_score, test_score

    return fit

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
CONJCORPUS_DIR = os.path.join(MODULE_DIR, '..', 'conjcorpus')
CACHE_DIR = os.path.join(MODULE_DIR, '..', 'cache')

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

    feature_extractors = [W2VExtractor(cache_file_name=os.path.join(CACHE_DIR, dataset + '_svmemb.pkl'))]

    logging.debug('loading feature...')
    X, y = feature_fuse(feature_extractors, sentences, labels)
    logging.debug('feature loaded')
    clf = CLF_CLASS()
    dump_file = os.path.join(MODEL_DIR, dataset + '_svm')
    try_params = dict(C=np.logspace(-1, 1, 8))
    if split is None:
        cv = StratifiedKFold(y, n_folds=args.fold, shuffle=True)
        clf = grid_search.GridSearchCV(clf, try_params, scoring='accuracy', cv=cv, verbose=1, n_jobs=20)
        clf.fit(X, y)
        print clf.best_score_
        print clf.grid_scores_
    else:
        train_ind, dev_ind, test_ind = (split=='train', split=='dev', split=='test')
        X_train, X_dev, X_test = X[train_ind], X[dev_ind], X[test_ind]
        y_train, y_dev, y_test = y[train_ind], y[dev_ind], y[test_ind]

        params = list(ParameterGrid(try_params))
        fit = fit_wrapper(X_train, y_train, X_dev, y_dev, X_test, y_test)
        res = Parallel(n_jobs=24)(delayed(fit)(**param) for param in params)
        res = sorted(res, key=lambda x: x[1], reverse=True)

        best_param, best_dev, best_test = res[0]

        print "Train/Dev/Test:{}/{}/{}".format(len(X_train), len(X_dev), len(X_test))
        print "Param: {}, Best score: {}".format(best_param, best_test)
