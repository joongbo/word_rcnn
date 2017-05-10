#!/usr/bin/python
# import sys
import _init_path

import timeit
import argparse
from collections import Counter

from datamanagement import *
from modelmanagement import *
from backpropagations import *
from activations import *

def main(fNames, opts, learning_opts):
    if fNames[0]=='baseModel': # if-statement for different model
        fresult = './savings/baseModel_' + fNames[3]
        from baseModel import building_model
    elif fNames[0]=='modelR':
        fresult = './savings/modelR_' + fNames[3]
        from modelR import building_model
    elif fNames[0]=='modelRC':
        fresult = './savings/modelRC_' + fNames[3]
        from modelRC import building_model
    else:
        raise NotImplementedError()
    dataFpath='../data/pickles/' + fNames[1]
    w2vFpath ='../data/pickles/' + fNames[2]
    
    print 'load data ...',
    data, vocab = load_sst(data_file=dataFpath)
    word2vec, word2idx = load_w2v(data_file=w2vFpath)
    assert len(vocab)==len(word2vec)
    print '\tdone.'
    
    # embedding
    opts['vocaSize'] = len(vocab)
    opts['w2v'] = word2vec
    # recurrent convolutional layer
    opts['maxLen'] = max(data[0][2])
    opts['maxLen_test'] = max(data[2][2])
    # multi-layer perceptron
    opts['outUnit'] = [len(Counter(data[0][1]).keys())]

    print 'fitting length of sentences ...',
    train_data = fit_length(data[0], max_len=opts['maxLen'])
    valid_data = fit_length(data[1], max_len=opts['maxLen_test'])
    test_data = fit_length(data[2], max_len=opts['maxLen_test'])
    print '\tdone.'

    print 'fitting mini-batch size for training data ...',
    train_data = fit_batch_add(train_data, opts['miniBatch'])
    print '\tdone.\n'
    
    start_time = timeit.default_timer()
    train_model, valid_model, test_model, layers, params = building_model(opts)
    end_time = timeit.default_timer()
    print 'Builing Model ran for %.2fm' % ( (end_time - start_time) / 60. )
    
    datasets = [train_data, valid_data, test_data]
    models = [train_model, valid_model, test_model]
    
    for i in xrange(learning_opts['trainN']):
        initializing_model(opts, layers, params)
        training_model(datasets, models, opts, learning_opts, fresult, params, i)

if __name__=='__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--modelFname', type=str, default='baseModel')
    p.add_argument('--dataFname', type=str, default='sst_word_mincut1_v15938.pkl')
    p.add_argument('--w2vFname', type=str, default='glove_sst_word_mincut1_v15938.pkl')
    p.add_argument('--logFname', type=str, default='sst_word_mincut1_v15938.pkl')
    #
    p.add_argument('--miniBatch', type=int, default=100)
    p.add_argument('--embdSize', type=int, default=300)
    p.add_argument('--embdUpdate', type=bool, default=True)
    p.add_argument('--borderMode', type=str, default='full')
    p.add_argument('--numStep', type=int, default=1)
    p.add_argument('--numFltrRC', type=int, default=50)
    p.add_argument('--fltrC', type=int, default=5)
    p.add_argument('--fltrR', type=int, default=5)
    p.add_argument('--pool', type=bool, default=True)
    p.add_argument('--poolMode', type=str, default='max')
    p.add_argument('--kTop', type=int, default=1)
    p.add_argument('--initSVD', type=bool, default=False)
    p.add_argument('--LRN', type=bool, default=True)
    p.add_argument('--dropRC', type=bool, default=True)
    p.add_argument('--dropFC', type=bool, default=True)
    p.add_argument('--dropRate', type=float, default=0.5)
    p.add_argument('--activationRC', default=relu)
    p.add_argument('--activationFC', default=tanh)
    p.add_argument('--L1', type=float, default=0.0)
    p.add_argument('--L2', type=float, default=0.0)
    p.add_argument('--updateRule', default=adam)
    #
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lrDecay', type=float, default=0.1)
    p.add_argument('--lrDecayStep', type=int, default=3)
    p.add_argument('--maxEpoch', type=int, default=10)
    p.add_argument('--trainN', type=int, default=3)
    args = p.parse_args()
    
    fNames = [args.modelFname, args.dataFname, args.w2vFname, args.logFname]
    
    opts = dict()
    # model 
    opts['miniBatch'] = args.miniBatch
    # embedding
    opts['embdSize'] = args.embdSize
    opts['embdUpdate'] = args.embdUpdate
    # recurrent convolutional layer
    opts['fltrRC'] = [[(args.numFltrRC, args.fltrC, args.fltrR)]]
    opts['borderMode'] = args.borderMode # one of ['valid', 'full', 'half']
    opts['numStep'] = args.numStep
    opts['activationRC'] = args.activationRC # [elu, relu, tanh, iden, sigm]
    opts['activationFC'] = args.activationFC # [elu, relu, tanh, iden, sigm]
    # pooling
    opts['pool'] = args.pool
    opts['poolMode'] = args.poolMode # one of ['d_k_max', 'k_max', 'max', 'spatial]
    opts['kTop'] = args.kTop
    # multi-layer perceptron
    # initializations
    opts['initSVD'] = args.initSVD
    # normalizations
    opts['LRN'] = args.LRN # local response normalization
    opts['dropRC'] = args.dropRC
    opts['dropFC'] = args.dropFC
    opts['dropRate'] = args.dropRate
    # generalizations
    opts['L1'] = args.L1 # L1-norm weight decay
    opts['L2'] = args.L2 # L2-norm weight decay
    # training parameters
    opts['updateRule'] = args.updateRule # one of ['adam', 'adadelta', 'adagrad', 'sgd', 'sgd_momentum']
    
    learning_opts = dict()
    # learning options
    learning_opts['lr'] = args.lr # initial learning rate
    learning_opts['lrDecay'] = args.lrDecay
    learning_opts['lrDecayStep'] = args.lrDecayStep
    learning_opts['maxEpoch'] = args.maxEpoch
    learning_opts['trainN'] = args.trainN
    
    main(fNames, opts, learning_opts)