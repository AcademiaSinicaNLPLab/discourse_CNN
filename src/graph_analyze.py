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
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical

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

from classifier.cnn import Kim_CNN
class Kim_CNN_Display(Kim_CNN):
    def __call__(self,
                 layers,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 nb_filter=100,
                 filter_length=[3,4,5],
                 nb_class=2):

        model = Sequential()
        model.add(self.get_emb_layer(vocabulary_size, embedding_dim, maxlen, layers[0].get_weights()))
        model.add(self.conv_Layer(maxlen, embedding_dim, nb_filter, filter_length, weights=[l.get_weights() for l in layers[1].layers]))
        model.add(TimeDistributed(Dense(nb_class, weights=weights[3])))
        model.add(Activation('softmax'))

        self.compile(model)
        return model

    def conv_Layer(self, maxlen, embedding_dim, nb_filter, filter_length, border_mode='same', weights=None):
        main_input = Input(shape=(maxlen, embedding_dim), name='main_input')
        convs = []
        for i, h in enumerate(filter_length):
            conv = Convolution1D(nb_filter=nb_filter,
                                 filter_length=h,
                                 border_mode=border_mode,
                                 activation='relu',
                                 subsample_length=1,
                                 input_shape=(maxlen, embedding_dim), weights=weights[1+i])(main_input)
            convs.append(conv)

        if len(convs)>1:
            output = merge(convs, mode='concat', concat_axis=-1)
        else:
            output = convs[0]
        return Model(main_input, output)

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

    m=1
    if m==1:
        cnn_extractor = CNNExtractor()
    elif m==2:
        cnn_extractor = ConjWeightCNNExtractor()

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

    cv = StratifiedKFold(y, n_folds=args.fold, shuffle=True)
    y = to_categorical(y)
    X_train = X
    y_train = y

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1)

    X_train = separateX(X_train)
    X_dev = separateX(X_dev)

    clf.fit(X_train, y_train,
            batch_size=50,
            nb_epoch=args.epoch,
            validation_data=(X_dev, y_dev),
            verbose=1)

    print
