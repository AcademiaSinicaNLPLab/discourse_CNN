#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')

if __name__ == '__main__':

    dfs = []
    datasets = sorted(sys.argv[1:])
    for dataset in datasets:
        df = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
        df['origin'] = dataset
        df['split'] = 'train'
        dfs.append(df)

    res = pandas.concat(dfs, ignore_index=True)

    res.to_pickle(os.path.join(CORPUS_DIR,'_'.join(datasets)+'.pkl'))
