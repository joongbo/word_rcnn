import _init_path
import numpy as np
import theano
import theano.tensor as T

from backpropagations import *
from activations import *
from WELayer import WELayer
from FCLayer import FCLayer
from CPLayer import CPLayer
from RCLayer import RCLayer
from FCdropLayer import FCdropLayer
from CPdropLayer import CPdropLayer
from RCdropLayer import RCdropLayer

def building_model(opts):
    print 'build layers of RCNN ...',
    rng = np.random.RandomState(21625)
    trng = T.shared_randomstreams.RandomStreams(21625)
    
    # allocate symbolic variables for the data
    x = T.imatrix('x') # the data is presented as rasterized images
    y = T.ivector('y') # the labels are presented as 1D vector of [int] labels
    s = T.ivector('s') # sentence lengths
    leaning_rate = T.fscalar('learning_rate')

    # embedding
    layerWE = WELayer(rng, V=opts['vocaSize'], k=opts['embdSize'])
    
    x_g = theano.gradient.grad_clip(x, -10, 10) # gradient clipping
    inputWords = layerWE.Words[x_g,:].dimshuffle(0,2,1,'x') # reshape input
    input_shape = (opts['miniBatch'], opts['embdSize'], opts['maxLen'], 1)

    fltrRC = np.asarray(opts['fltrRC'], dtype='int32')
    l_RC_layer, p_RC_layer, _ = fltrRC.shape
    inputs = []
    input_shapes = []
    for p in xrange(p_RC_layer):
        inputs.append(inputWords)
        input_shapes.append(input_shape)

    layers = [layerWE]
    for l in xrange(l_RC_layer):
        # convolution first
        players = []
        for p in xrange(p_RC_layer):
            layerCP = CPLayer(rng, trng, inputs[p], input_shapes[p], fltrRC[l][p], opts['borderMode'], LRN=opts['LRN'], 
                              activation=opts['activationRC'])
            players.append(layerCP)
        layers.append(players)

        # calculate for next layers
        inputs = []
        input_shapes = []
        for p in xrange(p_RC_layer):
            inputs.append(players[p].output)
            input_shapes.append(players[p].output_shape)

        # repeat above but using recurrent convolution
        players = []
        for p in xrange(p_RC_layer):
            if opts['dropRC']:
                layerRC = RCdropLayer(rng, trng, inputs[p], input_shapes[p], fltrRC[l][p], opts['numStep'], 
                                      pool=opts['pool'], pool_mode=opts['poolMode'], 
                                      L=l_RC_layer, l=l+1, k_top=opts['kTop'], s=s,
                                      LRN=opts['LRN'], p=opts['dropRate'], activation=opts['activationRC'])
            else:
                layerRC = RCLayer(rng, trng, inputs[p], input_shapes[p], fltrRC[l][p], opts['numStep'], 
                                  LRN=opts['LRN'], pool=opts['pool'], pool_mode=opts['poolMode'], 
                                  L=l_RC_layer, l=l+1, k_top=opts['kTop'], s=s,
                                  activation=opts['activationRC'])
            players.append(layerRC)
        layers.append(players)

        # calculate for next layers
        inputs = []
        input_shapes = []
        for p in xrange(p_RC_layer):
            inputs.append(players[p].output)
            input_shapes.append(players[p].output_shape)

    _input = T.concatenate(inputs, 1).flatten(2)
    _input_shapes = []
    for p in xrange(p_RC_layer):
        _input_shapes.append(input_shapes[p][1]*input_shapes[p][2])

    outUnit = np.asarray(opts['outUnit'])
    l_FC_layer = len(outUnit)

    outUnit = np.insert(outUnit, 0, sum(_input_shapes))
    for l in xrange(l_FC_layer):
        if l+1 == l_FC_layer:
            layerFC = FCLayer(rng, trng, _input, outUnit[l], outUnit[l+1],
                              activation=softmax)
        else:
            if opts['dropFC']:
                layerFC = FCdropLayer(rng, trng, _input, outUnit[l], outUnit[l+1],
                                      p=opts['dropRate'], activation=opts['activationFC'])
            else:
                layerFC = FCLayer(rng, trng, _input, outUnit[l], outUnit[l+1],
                                  activation=opts['activationFC'])
        _input = layerFC.output
        layers.append(layerFC)
    print '\tdone.'

    print 'build theano function of validation model ...',
    valid_model = theano.function([x, y, s], 
                                  layerFC.predict_errors(y), 
                                  allow_input_downcast=True, 
                                  on_unused_input='ignore')
    print '\tdone.'
    
    print 'build cost function and get update rule ...',
    # parameters
    params = []
    for layer in layers:
        if isinstance(layer, list):
            for _layer in layer:
                params += _layer.params
        else:
            params += layer.params

    norm_l1 = 0
    norm_l2 = 0
    for param in params:
        norm_l1 += T.sum(T.abs_(param))
        norm_l2 += T.sum(param**2)

    cost = layerFC.cross_entropy(y)
    #     cost = layerFC.negative_log_likelihood(y)
    cost += opts['L1']*norm_l1 + opts['L2']*norm_l2
    grads = T.grad(cost, params)
    updates = opts['updateRule'](grads, params, learning_rate = leaning_rate)
    print '\tdone.'

    print 'build theano function of training model ...',
    train_model = theano.function([x, y, s, leaning_rate], cost, 
                                  updates=updates, 
                                  allow_input_downcast=True, 
                                  on_unused_input='ignore')
    print '\tdone.'

    """ build test_model"""

    print 'build testing layers of RCNN ...',
    # allocate symbolic variables for the data
    x_test = T.ivector('x_test') # the data is presented as rasterized images
    y_test = T.iscalar('y_test') # the labels are presented as 1D vector of [int] labels
    s_test = T.iscalar('s_test') # sentence lengths

    # embedding
    inputWords_test = layerWE.Words[x_test,:].dimshuffle('x',1,0,'x') # reshape input
    input_shape_test = (1, opts['embdSize'], opts['maxLen_test'], 1)

    inputs = []
    input_shapes = []
    for p in xrange(p_RC_layer):
        inputs.append(inputWords_test)
        input_shapes.append(input_shape_test)

    layers_cnt = 1
    for l in xrange(l_RC_layer):
        # convolution first
        players = []
        for p in xrange(p_RC_layer):
            layerCP = CPLayer(rng, trng, inputs[p], input_shapes[p], fltrRC[l][p], opts['borderMode'], LRN=opts['LRN'], 
                              activation=opts['activationRC'], W=layers[layers_cnt][p].W, b=layers[layers_cnt][p].b)
            players.append(layerCP)
        layers_cnt += 1

        # calculate for next layers
        inputs = []
        input_shapes = []
        for p in xrange(p_RC_layer):
            inputs.append(players[p].output)
            input_shapes.append(players[p].output_shape)

        # repeat above but using recurrent convolution
        players = []
        for p in xrange(p_RC_layer):
            layerRC = RCLayer(rng, trng, inputs[p], input_shapes[p], fltrRC[l][p], opts['numStep'], 
                              LRN=opts['LRN'], pool=opts['pool'], pool_mode=opts['poolMode'], 
                              L=l_RC_layer, l=l+1, k_top=opts['kTop'], s=s,
                              activation=opts['activationRC'],W=layers[layers_cnt][p].W, b=layers[layers_cnt][p].b)
            players.append(layerRC)
        layers_cnt += 1

        # calculate for next layers
        inputs = []
        input_shapes = []
        for p in xrange(p_RC_layer):
            inputs.append(players[p].output)
            input_shapes.append(players[p].output_shape)

    _input = T.concatenate(inputs, 1).flatten(2)
    _input_shapes = []
    for p in xrange(p_RC_layer):
        _input_shapes.append(input_shapes[p][1]*input_shapes[p][2])

    outUnit = np.asarray(opts['outUnit'])
    l_FC_layer = len(outUnit)

    outUnit = np.insert(outUnit, 0, sum(_input_shapes))
    for l in xrange(l_FC_layer):
        if l+1 == l_FC_layer:
            layerFC = FCLayer(rng, trng, _input, outUnit[l], outUnit[l+1], 
                              activation=softmax, W=layers[layers_cnt].W, b=layers[layers_cnt].b)
        else:
            layerFC = FCLayer(rng, trng, _input, outUnit[l], outUnit[l+1],
                              activation=opts['activationFC'], W=layers[layers_cnt].W, b=layers[layers_cnt].b)
        layers_cnt += 1
        _input = layerFC.output
    print '\tdone.'

    print 'build theano function of testing model ...',
    test_model = theano.function([x_test, y_test, s_test], 
                                  layerFC.predict_errors(y_test.dimshuffle('x')), 
                                  allow_input_downcast=True, 
                                  on_unused_input='ignore')
    print '\tdone.'
    
    return train_model, valid_model, test_model, layers, params