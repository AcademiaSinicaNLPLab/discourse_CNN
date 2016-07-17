#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas
import argparse
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
CONJCORPUS_DIR = os.path.join(MODULE_DIR, '..', 'conjcorpus')

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', help='dataset name')
    return parser.parse_args(argv[1:])


conj_fol = [(w,) for w in ['but', 'however', 'nevertheless', 'otherwise', 'yet', 'still', 'nonetheless']]
conj_infer = [(w,) for w in ['therefore', 'furthermore', 'consequently', 'thus', 'subsequently', 'eventually', 'hence']]
conj_prev = [(w,) for w in ['till', 'until', 'despite', 'though', 'although', 'unless']]+[('even', 'if')]
conj_prev_head = [(w,) for w in ['while']]

mod = [(w,) for w in ['if', 'might', 'could', 'can', 'would', 'may']]
neg = [(w,) for w in ['n\'t', 'not', 'neither', 'never', 'no', 'nor']]

conj = conj_fol+conj_infer+conj_prev
All = conj+neg+mod

reverse_map = {}
reverse_map.update({marker:'conj_fol' for marker in conj_fol})
reverse_map.update({marker:'conj_infer' for marker in conj_infer})
reverse_map.update({marker:'conj_prev' for marker in conj_prev})
reverse_map.update({marker:'conj_prev_head' for marker in conj_prev_head})
reverse_map.update({marker:'mod' for marker in mod})
reverse_map.update({marker:'neg' for marker in neg})


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
                self.res = keyword
                break
        return self.res

def count_conj(sentences):
    counter = {marker:0 for marker in All+conj_prev_head}
    isIn = ConjChecker()
    for i, sentence in enumerate(sentences):
        wordarray = sentence.split()
        if len(wordarray)>0:
            marker = isIn(wordarray, 0, conj_prev_head)
            if marker:
                counter[marker]+=1
            for j in range(1, len(wordarray)):
                marker = isIn(wordarray, j, All)
                if marker:
                    counter[marker]+=1
    return counter

if __name__ == "__main__":
    
    args = parse_arg(sys.argv)
    dataset = args.dataset
    corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    counter = count_conj(corpus.sentence)
    counter = sorted([(k,v) for k,v in counter.items()], key=lambda x: x[1], reverse=True)
    group_counter = {group:0 for group in set(reverse_map.values())}
    for k, c in counter:
        print reverse_map[k], k, c
        group_counter[reverse_map[k]]+=c

    print 
    for k, v in group_counter.items():
        print k, v
