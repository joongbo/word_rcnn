import _init_path
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from backpropagations import *
from activations import *
from WordEmbdLayer import WordEmbdLayer
from FullConnLayer import FullConnLayer
from CuConvLayer import CuConvLayer
from FullConnLayerDrop import FullConnLayerDrop
from CuConvLayerDrop import CuConvLayerDrop

# define dropout
def dropout_from_layer(trng, input, p=0.5):
    ''' dropout '''
    mask = T.cast(trng.binomial(n=1, p=1-p, size=input.shape), theano.config.floatX)
    output = input * mask
    return output

def building_model(opts):
    print 'build layers of CuCNN ...'
    rng = np.random.RandomState(21625)
    trng = MRG_RandomStreams(21625)
    
    # allocate symbolic variables for the data
    x = T.imatrix('x') # the data is presented as rasterized images
    y = T.ivector('y') # the labels are presented as 1D vector of [int] labels
    s = T.ivector('s') # sentence lengths
    leaning_rate = T.fscalar('learning_rate')

    # embedding
    layerWordEmbd = WordEmbdLayer(rng, V=opts['vocaSize'], k=opts['embdSize'], w2v=opts['w2v'])
    
    x_g = theano.gradient.grad_clip(x, -10, 10) # gradient clipping
    inputWords = layerWordEmbd.Words[x_g,:].dimshuffle(0,2,1,'x') # reshape input
    if opts['dropWE']: inputWords = dropout_from_layer(trng, inputWords, p=opts['dropRate'])
    input_shape = (opts['miniBatch'], opts['embdSize'], opts['maxLen'], 1)

    fltrCuConv = np.asarray(opts['fltrRC'], dtype='int32')
    l_CuConv_layer, p_CuConv_layer, _ = fltrCuConv.shape
    inputs = []
    input_shapes = []
    for p in xrange(p_CuConv_layer):
        inputs.append(inputWords)
        input_shapes.append(input_shape)

    layers = [layerWordEmbd]
    for l in xrange(l_CuConv_layer):
        # repeat above but using recurrent convolution
        players = []
        if l+1 == l_CuConv_layer:
            for p in xrange(p_CuConv_layer):
                if opts['dropRC']:
                    layerCuConv = CuConvLayerDrop(rng, trng, inputs[p], input_shapes[p], fltrCuConv[l][p], opts['numStep'], 
                                          LRN=opts['LRN'], BN=opts['BN'], BN_mode=0,
                                          pool=True, pool_mode=opts['poolMode'], 
                                          L=l_CuConv_layer, l=l+1, k_top=opts['kTop'], s=s,
                                          p=opts['dropRate'], activation=opts['activationRC'])
                else:
                    layerCuConv = CuConvLayer(rng, trng, inputs[p], input_shapes[p], fltrCuConv[l][p], opts['numStep'], 
                                      LRN=opts['LRN'], BN=opts['BN'], BN_mode=0,
                                      pool=True, pool_mode=opts['poolMode'], 
                                      L=l_CuConv_layer, l=l+1, k_top=opts['kTop'], s=s,
                                      activation=opts['activationRC'])
                players.append(layerCuConv)
        else:
            for p in xrange(p_CuConv_layer):
                if opts['dropRC']:
                    layerCuConv = CuConvLayerDrop(rng, trng, inputs[p], input_shapes[p], fltrCuConv[l][p], opts['numStep'], 
                                          LRN=opts['LRN'], BN=opts['BN'], BN_mode=0,
                                          pool=opts['pool'], pool_mode=opts['poolMode'], 
                                          L=l_CuConv_layer, l=l+1, k_top=opts['kTop'], s=s,
                                          p=opts['dropRate'], activation=opts['activationRC'])
                else:
                    layerCuConv = CuConvLayer(rng, trng, inputs[p], input_shapes[p], fltrCuConv[l][p], opts['numStep'], 
                                      LRN=opts['LRN'], BN=opts['BN'], BN_mode=0,
                                      pool=opts['pool'], pool_mode=opts['poolMode'], 
                                      L=l_CuConv_layer, l=l+1, k_top=opts['kTop'], s=s,
                                      activation=opts['activationRC'])
                players.append(layerCuConv)
        layers.append(players)

        # calculate for next layers
        inputs = []
        input_shapes = []
        for p in xrange(p_CuConv_layer):
            inputs.append(players[p].output)
            input_shapes.append(players[p].output_shape)

    _input = T.concatenate(inputs, 1).flatten(2)
    _input_shapes = []
    for p in xrange(p_CuConv_layer):
        _input_shapes.append(input_shapes[p][1]*input_shapes[p][2])

    outUnit = np.asarray(opts['outUnit'])
    l_FullConn_layer = len(outUnit)

    outUnit = np.insert(outUnit, 0, sum(_input_shapes))
    for l in xrange(l_FullConn_layer):
        if l+1 == l_FullConn_layer:
            layerFullConn = FullConnLayer(rng, trng, _input, outUnit[l], outUnit[l+1], BN=opts['BN'], BN_mode=0,
                              activation=softmax)
        else:
            if opts['dropFC']:
                layerFullConn = FullConnLayerDrop(rng, trng, _input, outUnit[l], outUnit[l+1], BN=opts['BN'], BN_mode=0,
                                      p=opts['dropRate'], activation=opts['activationFC'])
            else:
                layerFullConn = FullConnLayer(rng, trng, _input, outUnit[l], outUnit[l+1], BN=opts['BN'], BN_mode=0,
                                  activation=opts['activationFC'])
        _input = layerFullConn.output
        layers.append(layerFullConn)
    
    print 'build cost function and get update rule ...'
    # parameters
    params = []
    for layer in layers:
        if isinstance(layer, list):
            for _layer in layer:
                params += _layer.params
        else:
            params += layer.params
    if opts['embdUpdate'] is not True:
        params = params[1:]
    
    # loss function
    norm_l1 = 0
    norm_l2 = 0
    for param in params:
        norm_l1 += T.sum(T.abs_(param))
        norm_l2 += T.sum(param**2)
    cost = layerFullConn.cross_entropy(y)
    # cost = layerFullConn.negative_log_likelihood(y)
    cost += opts['L1']*norm_l1 + opts['L2']*norm_l2
    
    grads = T.grad(cost, params)
    updates = opts['updateRule'](grads, params, learning_rate = leaning_rate)
    
    print 'build theano function of validation model ...'
    valid_model = theano.function([x, y, s], 
                                  layerFullConn.predict_errors(y), 
                                  allow_input_downcast=True, 
                                  on_unused_input='ignore')

    print 'build theano function of training model ...'
    train_model = theano.function([x, y, s, leaning_rate], cost, 
                                  updates=updates, 
                                  allow_input_downcast=True, 
                                  on_unused_input='ignore')

    """ build test_model"""

    print 'build testing layers of CuConvNN ...'
    # allocate symbolic variables for the data
    x_test = T.ivector('x_test') # the data is presented as rasterized images
    y_test = T.iscalar('y_test') # the labels are presented as 1D vector of [int] labels
    s_test = T.iscalar('s_test') # sentence lengths

    # embedding
    inputWords_test = layerWordEmbd.Words[x_test,:].dimshuffle('x',1,0,'x') # reshape input
    input_shape_test = (1, opts['embdSize'], opts['maxLen_test'], 1)

    inputs = []
    input_shapes = []
    for p in xrange(p_CuConv_layer):
        inputs.append(inputWords_test)
        input_shapes.append(input_shape_test)

    layers_cnt = 1
    for l in xrange(l_CuConv_layer):
        # repeat above but using recurrent convolution
        players = []
        if l+1 == l_CuConv_layer:
            for p in xrange(p_CuConv_layer):
                layerCuConv = CuConvLayer(rng, trng, inputs[p], input_shapes[p], fltrCuConv[l][p], opts['numStep'], 
                                  LRN=opts['LRN'], BN=opts['BN'], BN_mode=1,
                                  pool=True, pool_mode=opts['poolMode'], 
                                  L=l_CuConv_layer, l=l+1, k_top=opts['kTop'], s=s_test.dimshuffle('x'),
                                  activation=opts['activationRC'],W=layers[layers_cnt][p].W, b=layers[layers_cnt][p].b)
                players.append(layerCuConv)
        else:
            for p in xrange(p_CuConv_layer):
                layerCuConv = CuConvLayer(rng, trng, inputs[p], input_shapes[p], fltrCuConv[l][p], opts['numStep'], 
                                  LRN=opts['LRN'], BN=opts['BN'], BN_mode=1,
                                  pool=opts['pool'], pool_mode=opts['poolMode'], 
                                  L=l_CuConv_layer, l=l+1, k_top=opts['kTop'], s=s_test.dimshuffle('x'),
                                  activation=opts['activationRC'],W=layers[layers_cnt][p].W, b=layers[layers_cnt][p].b)
                players.append(layerCuConv)
        layers_cnt += 1

        # calculate for next layers
        inputs = []
        input_shapes = []
        for p in xrange(p_CuConv_layer):
            inputs.append(players[p].output)
            input_shapes.append(players[p].output_shape)

    _input = T.concatenate(inputs, 1).flatten(2)
    _input_shapes = []
    for p in xrange(p_CuConv_layer):
        _input_shapes.append(input_shapes[p][1]*input_shapes[p][2])

    outUnit = np.asarray(opts['outUnit'])
    l_FullConn_layer = len(outUnit)

    outUnit = np.insert(outUnit, 0, sum(_input_shapes))
    for l in xrange(l_FullConn_layer):
        if l+1 == l_FullConn_layer:
            layerFullConn = FullConnLayer(rng, trng, _input, outUnit[l], outUnit[l+1], BN=opts['BN'], BN_mode=1,
                              activation=softmax, W=layers[layers_cnt].W, b=layers[layers_cnt].b)
        else:
            layerFullConn = FullConnLayer(rng, trng, _input, outUnit[l], outUnit[l+1], BN=opts['BN'], BN_mode=1,
                              activation=opts['activationFC'], W=layers[layers_cnt].W, b=layers[layers_cnt].b)
        layers_cnt += 1
        _input = layerFullConn.output

    print 'build theano function of testing model ...'
    test_model = theano.function([x_test, y_test, s_test], 
                                  layerFullConn.predict_errors(y_test.dimshuffle('x')), 
                                  allow_input_downcast=True, 
                                  on_unused_input='ignore')
    
    return train_model, valid_model, test_model, layers, params, x