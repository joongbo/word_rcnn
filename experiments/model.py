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
    if fNames['model']=='modelCR': # if-statement for different model
        fresult = './savings/modelCR_' + fNames['log']
        from modelCR import building_model
    elif fNames['model']=='modelR':
        fresult = './savings/modelR_' + fNames['log']
        from modelR import building_model
    elif fNames['model']=='modelRC':
        fresult = './savings/modelRC_' + fNames['log']
        from modelRC import building_model
    else:
        raise NotImplementedError()
    dataFpath='../data/pickles/' + fNames['data']
    w2vFpath ='../data/pickles/' + fNames['w2v']
    
    print 'load data ...',
    data, vocab = load_sst(data_file=dataFpath)
    w2v, word2idx = load_w2v(data_file=w2vFpath)
    assert len(vocab)==len(w2v)
    
    # embedding
    opts['vocaSize'] = len(vocab)
    if opts['embd']:
        opts['w2v'] = w2v
    else:
        opts['w2v'] = None
    # recurrent convolutional layer
    opts['maxLen'] = max(data[0][2])
    opts['maxLen_test'] = max(data[2][2])
    # multi-layer perceptron
    if opts['outUnit'] is 0:
        opts['outUnit'] = [len(Counter(data[0][1]).keys())]
    else:
        opts['outUnit'] = [opts['outUnit'], len(Counter(data[0][1]).keys())]

    print 'fitting length of sentences ...',
    train_data = fit_length(data[0], max_len=opts['maxLen'])
    valid_data = fit_length(data[1], max_len=opts['maxLen_test'])
    test_data = fit_length(data[2], max_len=opts['maxLen_test'])

    print 'fitting mini-batch size for training data ...',
    train_data = fit_batch_add(train_data, opts['miniBatch'])
    print '\n'
    
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
    p.add_argument('--model', type=str, default='modelR')
    p.add_argument('--data', type=str, default='ssa_v18242.pkl')
    p.add_argument('--w2v', type=str, default='glove_ssa_v18242.pkl')
    p.add_argument('--log', type=str, default='ssa_v18242.pkl')
    #
    p.add_argument('--miniBatch', type=int, default=100)
    p.add_argument('--embd', type=int, default=1)
    p.add_argument('--embdSize', type=int, default=300)
    p.add_argument('--embdUpdate', type=int, default=1)
    p.add_argument('--borderMode', type=str, default='full')
    p.add_argument('--numStep', type=int, default=1)
    p.add_argument('--numFltrRC', type=int, default=50)
    p.add_argument('--fltrC', type=int, default=5)
    p.add_argument('--fltrR', type=int, default=5)
    p.add_argument('--pool', type=int, default=1)
    p.add_argument('--poolMode', type=str, default='max')
    p.add_argument('--kTop', type=int, default=1)
    p.add_argument('--outUnit', type=int, default=0)
    p.add_argument('--initSVD', type=int, default=0)
    p.add_argument('--LRN', type=int, default=1)
    p.add_argument('--dropRC', type=int, default=1)
    p.add_argument('--dropFC', type=int, default=1)
    p.add_argument('--dropRate', type=float, default=0.5)
    p.add_argument('--activationRC', default=relu)
    p.add_argument('--activationFC', default=tanh)
    p.add_argument('--L1', type=float, default=0.0)
    p.add_argument('--L2', type=float, default=0.0)
    p.add_argument('--updateRule', default=adam)
    #
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lrDecay', type=float, default=0.1)
    p.add_argument('--lrDecayStep', type=int, default=4)
    p.add_argument('--maxEpoch', type=int, default=12)
    p.add_argument('--trainN', type=int, default=3)
    args = p.parse_args()
    
    fNames = dict()
    fNames['model'] = args.model
    fNames['data'] = args.data
    fNames['w2v'] = args.w2v
    fNames['log'] = args.log
    
    opts = dict()
    # model 
    opts['miniBatch'] = args.miniBatch
    # embedding
    opts['embd'] = True if args.embd else False
    opts['embdSize'] = args.embdSize
    opts['embdUpdate'] = True if args.embdUpdate else False
    # recurrent convolutional layer
    opts['fltrRC'] = [[(args.numFltrRC, args.fltrC, args.fltrR)]]
    opts['borderMode'] = args.borderMode # one of ['valid', 'full', 'half']
    opts['numStep'] = args.numStep
    opts['activationRC'] = args.activationRC # [elu, relu, tanh, iden, sigm]
    opts['activationFC'] = args.activationFC # [elu, relu, tanh, iden, sigm]
    # pooling
    opts['pool'] = True if args.pool else False
    opts['poolMode'] = args.poolMode # one of ['d_k_max', 'k_max', 'max', 'spatial]
    opts['kTop'] = args.kTop
    # multi-layer perceptron
    opts['outUnit'] = args.outUnit
    # initializations
    opts['initSVD'] = True if args.initSVD else False
    # normalizations
    opts['LRN'] = True if args.LRN else False # local response normalization
    opts['dropRC'] = True if args.dropRC else False
    opts['dropFC'] = True if args.dropFC else False
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
    
    print '\nfile names', fNames
    print 'hyper parameters', opts
    print 'learning parameters', learning_opts, '\n'
    
    main(fNames, opts, learning_opts)