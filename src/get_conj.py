#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas
import argparse
import os
from feature.extractor import feature_fuse, W2VExtractor, CNNExtractor

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', help='dataset name')
    return parser.parse_args(argv[1:])

conj_fol = [(w,) for w in ['but', 'however', 'nevertheless', 'otherwise', 'yet', 'still', 'nonetheless']]
conj_infer = [(w,) for w in ['therefore', 'furthermore', 'consequently', 'thus', 'subsequently', 'eventually', 'hence']]
conj_fol = conj_fol+conj_infer
conj_prev = [(w,) for w in ['till', 'until', 'despite', 'though', 'although', 'unless']]+[('even', 'if')]
conj_prev_head = [(w,) for w in ['while']]

mod = [(w,) for w in ['if', 'might', 'could', 'can', 'would', 'may']]
neg = [(w,) for w in ['n\'t', 'not', 'neither', 'never', 'no', 'nor']]

conj = conj_fol+conj_prev
All = conj+neg

class ConjChecker(object):
    def __init__(self):
        self.res = 0

    def __call__(self, wordarray, i, category):
        self.res = 0
        for keyword in category:
            equal_one = True
            for j, key in enumerate(keyword):
                if i+j==len(wordarray) or wordarray[i+j] != key:
                    equal_one = False
                    break
            if equal_one:
                self.res = len(keyword)
                break
        return self.res

def get_conj_ind(sentences, inv=False):
    res = []
    inv_res = []
    isIn = ConjChecker()
    for i, sentence in enumerate(sentences):
        isin = False 
        wordarray = sentence.split()
        for j in range(len(wordarray)):
            if isIn(wordarray, j, All):
                isin = True
                break
        if isin:
            res.append(i)
        else:
            inv_res.append(i)
    if inv:
        return inv_res
    else:
        return res

if __name__ == "__main__":
    
    args = parse_arg(sys.argv)
    dataset = args.dataset
    corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    conj_ind = get_conj_ind(corpus.sentence)

    corpus.loc[conj_ind].to_csv(os.path.join(CORPUS_DIR, 'data', 'CONJ'+dataset+'.all'), index=False, doublequote=False, sep=' ', header=False, columns=['label','sentence'])
    corpus.loc[conj_ind].to_pickle(os.path.join(CORPUS_DIR, 'CONJ'+dataset+'.pkl'))

    inv_conj_ind = get_conj_ind(corpus.sentence, inv=True)

    corpus.loc[inv_conj_ind].to_csv(os.path.join(CORPUS_DIR, 'data', 'INVCONJ'+dataset+'.all'), index=False, doublequote=False, sep=' ', header=False, columns=['label','sentence'])
    corpus.loc[inv_conj_ind].to_pickle(os.path.join(CORPUS_DIR, 'INVCONJ'+dataset+'.pkl'))
