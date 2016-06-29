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
logging.basicConfig(level=logging.DEBUG)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
MODEL_DIR = os.path.join(MODULE_DIR, '..', 'model')

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', help='dataset name')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-cv', '--fold', type=int, default=10, help='K fold')
    return parser.parse_args(argv[1:])

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    np.random.seed(args.seed)

    dataset = args.dataset
    corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    sentences, labels = list(corpus.sentence), list(corpus.label)

    if len(set(corpus.split.values))==1:
        split = None
    else:
        split = corpus.split.values

    feature_extractors = [W2VExtractor()]
    logging.debug('loading feature...')
    X, y = feature_fuse(feature_extractors, sentences, labels)
    logging.debug('feature loaded')
    clf = LinearSVC()
    dump_file = os.path.join(MODEL_DIR, dataset + '_svm')
    if split is None:
        OVA = True
        parameters = dict(C=np.logspace(-5, 1, 8))

        cv = StratifiedKFold(y, n_folds=args.fold, shuffle=True)
        clf = grid_search.GridSearchCV(clf, parameters, scoring='accuracy', cv=cv, verbose=1, n_jobs=20)
        clf.fit(X, y)
        print clf.best_score_
    else:
        assert False, "No split not implemented yet"
        raise NotImplementedError
