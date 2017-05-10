#!/usr/bin/python
# import sys
import _init_path

from datamanagement import *
from modelmanagement import *
from backpropagations import *
from activations import *

import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu1")

def main():
    data_fname='sst_word_mincut1_voca15938.pkl'
    data_fpath='../data/pickles/' + data_fname
    w2v_fpath ='../data/pickles/w2v_sst_glove_mincut1_voca15938.pkl'
    
    if True: # [Not implemented] if-statement for different model
        fresult = './savings/baseModel_' + data_fname
        from baseModel import building_model # 'baseModel' can be replaced
    
    print 'load data ...',
    data, vocab = load_sst(data_file=data_fpath)
    word2vec, word2idx = load_w2v(data_file=w2v_fpath)
    assert len(vocab)==len(word2vec)
    print '\tdone.'

    opts = dict()
    # model 
    opts['miniBatch'] = 100
    # embedding
    opts['vocaSize'] = len(vocab)
    opts['embdSize'] = 300
    opts['w2v'] = word2vec
    opts['embdUpdate'] = True
    # recurrent convolutional layer
    opts['maxLen'] = max(data[0][2])
    opts['maxLen_test'] = max(data[2][2])
    opts['fltrRC'] = [[(300,5,5)]]
    opts['borderMode'] = 'full' # one of ['valid', 'full', 'half']
    opts['numStep'] = 1
    opts['activationRC'] = relu # [elu, relu, tanh, iden, sigm]
    opts['activationFC'] = tanh # [elu, relu, tanh, iden, sigm]
    # pooling
    opts['pool'] = True
    opts['poolMode'] ='max' # one of ['d_k_max', 'k_max', 'max', 'spatial]
    opts['kTop'] = 1
    # multi-layer perceptron
    opts['outUnit'] = [5]
    # initializations
    opts['initSVD'] = False
    # normalizations
    opts['LRN'] = True # local response normalization
    opts['dropRC'] = True
    opts['dropFC'] = True
    opts['dropRate'] = 0.8
    # generalizations
    opts['L1'] = 0 # L1-norm weight decay
    opts['L2'] = 0 # L2-norm weight decay
    # training parameters
    opts['updateRule'] = adam # one of ['adam', 'adadelta', 'adagrad', 'sgd', 'sgd_momentum']

    learning_opts = dict()
    # learning options
    learning_opts['lr'] = 0.001 # initial learning rate
    learning_opts['lrDecay'] = 0.5
    learning_opts['lrDecayStep'] = 5
    learning_opts['maxEpoch'] = 15

    print 'fitting length of sentences ...',
    train_data = fit_length(data[0], max_len=opts['maxLen'])
    valid_data = fit_length(data[1], max_len=opts['maxLen_test'])
    test_data = fit_length(data[2], max_len=opts['maxLen_test'])
    print '\tdone.'

    print 'fitting mini-batch size for training data ...',
    train_data = fit_batch_add(train_data, opts['miniBatch'])
    print '\tdone.\n'
    
    train_model, valid_model, test_model, layers, params = building_model(opts)
    datasets = [train_data, valid_data, test_data]
    models = [train_model, valid_model, test_model]
    
    opts['initSVD'] = True
    for i in xrange(1):
        initializing_model(opts, layers, params)
        training_model(datasets, models, opts, learning_opts, fresult, params, i)
        
    opts['initSVD'] = False
    for i in xrange(1):
        initializing_model(opts, layers, params)
        training_model(datasets, models, opts, learning_opts, fresult, params, i+5)

if __name__=='__main__':
    main()