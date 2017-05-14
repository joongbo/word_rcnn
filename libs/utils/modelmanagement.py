import timeit
from collections import OrderedDict
import cPickle as pkl

import numpy as np
import theano

from datamanagement import *
from backpropagations import *

def training_model(datasets, models, opts, learning_opts, fresult, params, i=0):
    train_data, valid_data, test_data = datasets
    train_model, valid_model, test_model = models
    fresult = fresult.replace('.pkl', '_trial' + str(i) +'.pkl')
    fpresult = fresult.replace('savings','logs')
    fpresult = fpresult.replace('pkl','txt')
    mb = opts['miniBatch']
    n_mb_train = len(train_data[1])
    n_mb_train //= mb
    
    n_valid = len(valid_data[1])
    n_test = len(test_data[1])
        
    lr = learning_opts['lr']
    lrDecay = learning_opts['lrDecay']
    lrDecayStep = learning_opts['lrDecayStep']
    
    check_iter = 10
    if n_mb_train > 100:
        check_iter = 100
    elif n_mb_train > 1000:
        check_iter = 1000
            
    best_valid_accuracy = 0
    start_time = timeit.default_timer()
    fp = open(fpresult, 'wb')
    fp.write(str(opts))
    fp.write(str(learning_opts))
    print 'training ...'
    for epoch in xrange(learning_opts['maxEpoch']):
        train_x, train_y, train_s = shuffle_data(train_data)
        
        for mb_index in xrange(n_mb_train):
            iter = epoch * n_mb_train + mb_index
            if (iter+1) % check_iter == 0:
                print '\t[iter] %i' %(iter+1),
                print '[time] %.2fm' %((timeit.default_timer()-start_time)/60.),
                print '[loss] %.4f ' %(loss)
            loss = train_model(train_x[mb_index * mb: (mb_index + 1) * mb],
                               train_y[mb_index * mb: (mb_index + 1) * mb],
                               train_s[mb_index * mb: (mb_index + 1) * mb], 
                               lr)
            if opts['embdUpdate']:
                # normalize embeddings and set zeros for padding index
                matrix_embd = params[0].get_value()
                matrix_embd[0] = np.zeros(opts['embdSize'])
                matrix_embd[1:] /= np.sqrt( (matrix_embd[1:]**2).sum(axis=1) )[:,None]
                params[0].set_value(matrix_embd)
                
            
        train_errors = []
        for mb_index in xrange(n_mb_train):
            _error = valid_model(train_x[mb_index * mb: (mb_index + 1) * mb],
                                 train_y[mb_index * mb: (mb_index + 1) * mb],
                                 train_s[mb_index * mb: (mb_index + 1) * mb] )
            train_errors.append(_error)
        train_accuracy = 1 - np.mean(train_errors)
        print '\ttrain performance (accuracy) %.2f %%' %(train_accuracy*100.)
        fp.write('\ttrain performance (accuracy) %.2f %%\n' %(train_accuracy*100.))
        
        # compute error on validation set
        now_valid_accuracy = testing_model(valid_data, test_model)

        print '\t[epoch] %i' %(epoch+1),
        print 'minibatch %i/%i' %(mb_index+1,n_mb_train),
        print 'validation performance (accuracy) %.2f %%' %(now_valid_accuracy*100.)
        fp.write('\t[epoch] %i minibatch %i/%i validation performance (accuracy) %.2f %%\n' 
                 %(epoch+1, mb_index+1, n_mb_train, now_valid_accuracy*100.))
        
        # if we got the best validation score until now
        if now_valid_accuracy > best_valid_accuracy:
            f = open(fresult, 'wb')
            pkl.dump([opts, learning_opts, params], f, -1)
            f.close()
            # save best validation score and iteration number
            best_valid_accuracy = now_valid_accuracy

            # test it on the test set
            test_accuracy = testing_model(test_data, test_model)
            print '\ttest performance of best model (accuracy) %.2f %%' %(test_accuracy*100.)
            fp.write('\ttest performance of best model (accuracy) %.2f %%\n' 
                     %(test_accuracy*100.))
            
        if (epoch+1) % lrDecayStep == 0:
            f = open(fresult, 'rb')
            _, _, new_params = pkl.load(f)
            f.close()
            for param, new_param in zip(params, new_params):
                param.set_value(new_param.get_value())
            lr *= lrDecay

            
    end_time = timeit.default_timer()
    print 'Optimization complete.'
    print 'Best validation score of %f %% with test performance %f %%' %(best_valid_accuracy * 100., test_accuracy * 100.) 
    print 'The code ran for %.2fm' % ( (end_time - start_time) / 60. )
    fp.write('Best validation score of %f %% with test performance %f %%\n' 
             %(best_valid_accuracy * 100., test_accuracy * 100.))
    fp.write('The code ran for %.2fm\n' 
             %((end_time - start_time)/60.))
    fp.close()

    
def testing_model(data, model):
    n_data = len(data[1])

    data_x,  data_y,  data_s  = data

    data_errors = []
    for _index in xrange(n_data):
        _errors = model( data_x[_index], data_y[_index], data_s[_index] )
        data_errors.append(_errors)
    accuracy = 1 - np.mean(data_errors)
    return accuracy


def initializing_model(opts, layers, params):
    new_params = []
    cnt = 0
    for layer in layers:
        if isinstance(layer, list):
            for _layer in layer:
                _layer.initialize_weights(svd_init=opts['initSVD'])
                new_params += _layer.params
                if opts['BN']:
                    _layer._batch_normalize.initialize_weights()
                    new_params += _layer._batch_normalize.params
        else:
            layer.initialize_weights(svd_init=opts['initSVD'])
            new_params += layer.params
            if opts['BN'] and cnt != 0:
                layer._batch_normalize.initialize_weights()
                new_params += layer._batch_normalize.params
        cnt += 1
        
    if opts['embdUpdate'] is not True:
        new_params = new_params[1:]
    #print "params:", params
    #print "new_params:", new_params
    for param, new_param in zip(params, new_params):
        param.set_value(new_param.get_value())
    
    print 'Initialized model successfully'
