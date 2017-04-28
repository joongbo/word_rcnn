import numpy as np

import theano
from theano import scan
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


""" Activations """
def elu(x): # exponential linear unit
    y = T.nnet.elu(x)
    return y

def relu(x): # recifier linear unit
    y = T.nnet.relu(x)
    return y

def sigm(x): # sigmoid
    y = T.nnet.sigmoid(x)
    return y

def tanh(x): # tanh
    y = T.tanh(x)
    return y

def iden(x): # identity
    y = x
    return y

def softmax(x):
    y = T.nnet.softmax(x)
    return y

def _svd_orthonormal(rng, shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = rng.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q

def _dropout_from_layer(trng, layer, p):
    """p is the probablity of dropping a unit
    """
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = T.cast(trng.binomial(n=1, p=1-p, size=layer.shape), theano.config.floatX)
    # The cast is important because int * float32 = float64 which pulls things off the gpu
    output = layer * mask
    return output



""" Layers (Classes) """
class WE_layer(object):
    """ Layer of a word embedding """
    def __init__(self, rng, voca_size, embd_size):
        self.rng = rng
        self.voca_size = voca_size
        self.embd_size = embd_size
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        _value = self.rng.randn(self.voca_size, self.embd_size) / np.sqrt(self.voca_size)
        embedding = theano.shared(name='embedding', value=_value.astype(theano.config.floatX), borrow=True)
        self.embedding = embedding
        self.params = [self.embedding]

class FC_layer(object):
    """ Layer of a fully connected (or feed forward) network """
    def __init__(self, rng, trng, input, n_in, n_out,
                 W=None, b=None, activation=tanh):
        
        self.rng = rng
        self.trng = trng
        self.input = input
        self.mini_batch = input.shape[0]
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if W is None and b is None:
            self._initialize_weights()
        else:
            if W is None or b is None:
                raise TypeError('one of parameters is not given', ('W', W, 'b', b)) 
            self.W = W
            self.b = b
            self.params = [self.W, self.b]

            

        # parameters of the model
        self.loutput = T.dot(self.input, self.W) + self.b
        self.output = self.loutput
        self.output_shape = (input.shape[0], n_out)

        self.output = self.activation(self.output)
        
        
    def _initialize_weights(self):
        if self.activation in [relu]:
            W_value = self.rng.randn(self.n_in, self.n_out) / np.sqrt(self.n_in/2)
        else:
            W_value = self.rng.randn(self.n_in, self.n_out) / np.sqrt(self.n_in)

        b_value = np.zeros((self.n_out,), dtype=theano.config.floatX)
        W = theano.shared(name='W', value=W_value.astype(theano.config.floatX), borrow=True)
        b = theano.shared(value=b_value, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]

    # classifier
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def cross_entropy(self, y):
        return T.mean(T.nnet.categorical_crossentropy(self.output, y))

    def predict_errors(self, y):
        y_pred = T.argmax(self.output, axis=1)
        if y.ndim != y_pred.ndim: # check if y has same dimension of output
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', y_pred.type))
        if y.dtype.startswith('int'): # check if y is of the correct datatype
            return T.mean( T.neq( y_pred, y ) ) # T.neq: not equal
        else:
            raise NotImplementedError()
        
        
class CP_layer(object):
    """ Layer of a convolutional network with pooling options"""
    def __init__(self, rng, trng, input, input_shape, filter_figure, border_mode, 
                 pool=False, pool_mode='k_max', L=None, l=None, k_top=None, s=None, LRN=False, 
                 W=None, b=None, activation=tanh):
        
        # check variables and calculate dependancies
        if input.ndim!=4:
            raise TypeError('input dimension must be 4, given', input.ndim)    
        
        filter_shape = (filter_figure[0], input_shape[1], filter_figure[1], input_shape[3])
        if border_mode=='valid': output_border = input_shape[2] - filter_shape[2] + 1
        elif border_mode=='half':
            if filter_figure[1]%2==0:
                raise TypeError('filter must be odd with "half" mode')
            output_border = input_shape[2]
        elif border_mode == 'full': output_border = input_shape[2] + filter_shape[2] - 1
        else: raise TypeError('border_mode must be one of [valid, half, full], given', border_mode)
        output_shape = (input_shape[0], filter_shape[0], output_border, input_shape[3])

        if pool:
            if not (pool_mode is 'k_max' or pool_mode is 'd_k_max' or 
                    pool_mode is 'spatial' or pool_mode is 'max'):
                raise TypeError('pool_mode must be one of [k_max, d_k_max, max, spatial], given', pool_mode)
            if L is None or l is None or k_top is None or s is None:
                raise TypeError('any pooling variables must not be None')
                
        # assign class variables
        self.rng = rng
        self.trng = trng
        self.input = input
        self.input_shape = input_shape
        self.activation = activation
        
        self.filter_shape = filter_shape
        self.border_mode = border_mode
        self.output_shape = output_shape
        
        self.pool = pool
        self.pool_mode = pool_mode
        self.s = s
        self.k_top = k_top
        self.L = L
        self.l = l

        self.LRN = LRN
        
        if W is None and b is None:
            self._initialize_weights()
        else:
            if W is None or b is None:
                raise TypeError('one of parameters is not given', ('W', W, 'b', b)) 
            self.W = W
            self.b = b
            self.params = [self.W, self.b]
            
        # convolution output (feed forward)
        conv_out = conv2d(input=self.input, input_shape=self.input_shape, 
                          filters=self.W, filter_shape=self.filter_shape, 
                          border_mode=self.border_mode)
        self.output = conv_out + self.b.dimshuffle('x',0,'x','x')
        
        self.output = self.activation(self.output) # activation
        
        if self.LRN: self.local_resp_normalize(self.filter_shape[0])
            
        if self.pool:
            if self.pool_mode is 'd_k_max':
                self.dynamic_k_max_pooling()
            elif self.pool_mode=='k_max':
                self.k_max_pooling()
            elif self.pool_mode=='max' or self.pool_mode=='spatial':
                self.pooling()
                
                
    def _initialize_weights(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
        W_value = self.rng.randn(self.filter_shape[0],self.filter_shape[1],self.filter_shape[2],self.filter_shape[3])
        if self.activation in [relu]:
            W_value = W_value / np.sqrt(fan_in / 2)
        else:
            W_value = W_value / np.sqrt(fan_in)
        b_value = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        W = theano.shared(name='W', value=W_value.astype(theano.config.floatX), borrow=True)
        b = theano.shared(value=b_value, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        
    # response normalization
    def local_resp_normalize(self, K, alpha=0.001, beta=0.75):
        N = K/8 + 1
        squared = self.output ** 2
        outputs = []
        for k in xrange(K):
            s = T.maximum(0, k-N//2)
            e = T.minimum(K, k+N//2)
            tmp_out = self.output[:,k,:,:] / ( (1. + alpha/N*( T.sum(squared[:,s:e,:,:], axis=1) ) )**beta)
            outputs.append( tmp_out.dimshuffle(0,'x',1,2) )
            
        self.output = T.concatenate(outputs, 1)
        
    # pooling methods
    def pooling(self):
        input = self.output
        input_shape = self.output_shape
        
        s = float(input_shape[2])
        k_l = max(self.k_top, int(s*(self.L-self.l)//self.L))
        pool_shape = (int( s/k_l ), 1)
        
        self.output = pool.pool_2d(input=input, ds=pool_shape, ignore_border=True)
        self.output_shape = (input_shape[0], input_shape[1], k_l, input_shape[3])

    def k_max_pooling(self):
        input = self.output
        input_shape = self.output_shape
        
        s = input_shape[2]
        k_l = max(self.k_top, s*(self.L-self.l)//self.L)
        sorted_idx = T.argsort(input, axis = 2)
        sorted_idx = sorted_idx[:,:,-k_l:,:]
        sorted_idx  = T.sort(sorted_idx, axis=2)

        output = input.reshape((input_shape[0]*input_shape[1],input_shape[2]))
        ii = T.repeat(T.arange(output.shape[0]), sorted_idx.shape[2])
        jj = sorted_idx.flatten()

        self.output = output[ii,jj].reshape(sorted_idx.shape)
        self.output_shape = (input_shape[0], input_shape[1], k_l, input_shape[3])

    def dynamic_k_max_pooling(self):
        input = self.output
        input_shape = self.output_shape
        
        reduced = max(self.k_top, input_shape[2]*(self.L-self.l)//self.L)
        k_l = T.maximum(self.k_top, self.s*(self.L-self.l)//self.L)
        input_shape = np.asarray(input_shape, dtype='int32')
        def pooling_step(x, k, input_shape):
            reshaped = x.reshape((input_shape[1],input_shape[2])) # reshaped from 3-dim to 2-dim
            sorted_idx = T.argsort(reshaped, axis = 1)
            sorted_idx = sorted_idx[:,-k:]
            sorted_idx  = T.sort(sorted_idx, axis=1)

            ii = T.repeat(T.arange(reshaped.shape[0]), sorted_idx.shape[1])
            jj = sorted_idx.flatten()

            sorted_img = reshaped[ii,jj].reshape(sorted_idx.shape) # flat to 2-dim
            padding = T.zeros((sorted_img.shape[0], reduced - k), dtype=theano.config.floatX)
            padded = T.concatenate([sorted_img, padding], axis=1)
            return padded

        output, _ = theano.scan(fn = pooling_step, 
                                sequences = [input, k_l], 
                                non_sequences = input_shape) 
        
        self.output_shape = (input_shape[0], input_shape[1], reduced, input_shape[3])
        self.output = output.reshape(self.output_shape)
            
            
class RC_layer(object):
    """ Layer of a recurrent convolutional network """
    def __init__(self, rng, trng, input, input_shape, filter_figure, n_steps, 
                 pool=False, pool_mode='k_max', L=None, l=None, k_top=None, s=None, LRN=False, 
                 W=None, b=None, activation=tanh):
        
        # check variables and calculate dependancies
        if input.ndim!=4:
            raise TypeError('input dimension must be 4, given', input.ndim)
        if filter_figure[2]%2==0:
            raise TypeError('recurrent filter must be odd with "half" mode')
        if filter_figure[0]!=input_shape[1]:
            raise TypeError('recurrent filter must have same feature maps with input channel')
            
        output_border = input_shape[2]
        filter_shape = (filter_figure[0], filter_figure[0], filter_figure[2], input_shape[3])
        output_shape = (input_shape[0], filter_shape[0], output_border, input_shape[3])
        
        if pool:
            if not (pool_mode is 'k_max' or pool_mode is 'd_k_max' or 
                    pool_mode is 'spatial' or pool_mode is 'max'):
                raise TypeError('pool_mode must be one of [k_max, d_k_max, max, spatial], given', pool_mode)
            if L is None or l is None or k_top is None or s is None:
                raise TypeError('any pooling variables must not be None')
                
        # assign class variables
        self.rng = rng
        self.trng = trng
        self.input = input
        self.input_shape = input_shape
        self.activation = activation
        
        self.filter_shape = filter_shape
        self.output_shape = output_shape
        self.n_steps = n_steps
        
        self.pool = pool
        self.pool_mode = pool_mode
        self.s = s
        self.k_top = k_top
        self.L = L
        self.l = l

        self.LRN = LRN
        if W is None and b is None:
            self._initialize_weights()
        else:
            if W is None or b is None:
                raise TypeError('one of parameters is not given', ('W', W, 'b', b)) 
            self.W = W
            self.b = b
            self.params = [self.W, self.b]
        
        
        # recurrent convolution output
        def one_step(out_tm1):
            conv_out = conv2d(input=out_tm1, input_shape=self.input_shape, 
                           filters=self.W, filter_shape=self.filter_shape, 
                           border_mode='half')
            out_t = self.input + conv_out + self.b.dimshuffle('x',0,'x','x')
            out_t = self.activation(out_t)
            if self.LRN: out_t = self.local_resp_normalize(out_t, self.filter_shape[0])
                    
            return out_t
        
        self.initial = T.unbroadcast(self.input, 3)
        out_t, _ = scan(fn=one_step,
                        outputs_info=dict(initial=self.initial),
                        n_steps=self.n_steps)
        self.output = out_t[-1]
                
        if self.pool:
            if self.pool_mode is 'd_k_max':
                self.dynamic_k_max_pooling()
            elif self.pool_mode=='k_max':
                self.k_max_pooling()
            elif self.pool_mode=='max' or self.pool_mode=='spatial':
                self.pooling()
        
    def _initialize_weights(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
        W_value = self.rng.randn(self.filter_shape[0],self.filter_shape[1],self.filter_shape[2],self.filter_shape[3])
        if self.activation in [relu]:
            W_value = W_value / np.sqrt(fan_in / 2)
        else:
            W_value = W_value / np.sqrt(fan_in)
        b_value = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        W = theano.shared(name='W', value=W_value.astype(theano.config.floatX), borrow=True)
        b = theano.shared(value=b_value, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        
    # response normalization
    def local_resp_normalize(self, input, K, alpha=0.001, beta=0.75):
        N = K/8 + 1
        squared = input ** 2
        outputs = []
        for k in xrange(K):
            s = T.maximum(0, k-N//2)
            e = T.minimum(K, k+N//2)
            tmp_out = input[:,k,:,:] / ( (1. + alpha/N*( T.sum(squared[:,s:e,:,:], axis=1) ) )**beta)
            outputs.append( tmp_out.dimshuffle(0,'x',1,2) )
            
        return T.concatenate(outputs, 1)
        
    # pooling methods
    def pooling(self):
        input = self.output
        input_shape = self.output_shape
        
        s = float(input_shape[2])
        k_l = max(self.k_top, int(s*(self.L-self.l)//self.L))
        pool_shape = (int( s/k_l ), 1)
        
        self.output = pool.pool_2d(input=input, ds=pool_shape, ignore_border=True)
        self.output_shape = (input_shape[0], input_shape[1], k_l, input_shape[3])

    def k_max_pooling(self):
        input = self.output
        input_shape = self.output_shape
        
        s = input_shape[2]
        k_l = max(self.k_top, s*(self.L-self.l)//self.L)
        sorted_idx = T.argsort(input, axis = 2)
        sorted_idx = sorted_idx[:,:,-k_l:,:]
        sorted_idx  = T.sort(sorted_idx, axis=2)

        output = input.reshape((input_shape[0]*input_shape[1],input_shape[2]))
        ii = T.repeat(T.arange(output.shape[0]), sorted_idx.shape[2])
        jj = sorted_idx.flatten()

        self.output = output[ii,jj].reshape(sorted_idx.shape)
        self.output_shape = (input_shape[0], input_shape[1], k_l, input_shape[3])

    def dynamic_k_max_pooling(self):
        input = self.output
        input_shape = self.output_shape
        
        reduced = max(self.k_top, input_shape[2]*(self.L-self.l)//self.L)
        k_l = T.maximum(self.k_top, self.s*(self.L-self.l)//self.L)
        input_shape = np.asarray(input_shape, dtype='int32')
        def pooling_step(x, k, input_shape):
            reshaped = x.reshape((input_shape[1],input_shape[2])) # reshaped from 3-dim to 2-dim
            sorted_idx = T.argsort(reshaped, axis = 1)
            sorted_idx = sorted_idx[:,-k:]
            sorted_idx  = T.sort(sorted_idx, axis=1)

            ii = T.repeat(T.arange(reshaped.shape[0]), sorted_idx.shape[1])
            jj = sorted_idx.flatten()

            sorted_img = reshaped[ii,jj].reshape(sorted_idx.shape) # flat to 2-dim
            padding = T.zeros((sorted_img.shape[0], reduced - k), dtype=theano.config.floatX)
            padded = T.concatenate([sorted_img, padding], axis=1)
            return padded

        output, _ = theano.scan(fn = pooling_step, 
                                sequences = [input, k_l], 
                                non_sequences = input_shape) 
        
        self.output_shape = (input_shape[0], input_shape[1], reduced, input_shape[3])
        self.output = output.reshape(self.output_shape)


class DR_FC_layer(FC_layer):
    def __init__(self, rng, trng, input, n_in, n_out,
                 dropout_rate=0.5, W=None, b=None, activation=tanh):
        super(DR_FC_layer, self).__init__(
            rng=rng, trng=trng, input=input, n_in=n_in, n_out=n_out, 
            W=W, b=b, activation=activation)

        self.output = _dropout_from_layer(trng, self.output, p=dropout_rate)
        
class DR_RC_layer(RC_layer):
    """ Layer of a recurrent convolutional network """
    def __init__(self, rng, trng, input, input_shape, filter_figure, n_steps, 
                 pool=False, pool_mode='k_max', L=None, l=None, k_top=None, s=None, LRN=False, 
                 dropout_rate=0.5, W=None, b=None, activation=tanh):
        super(DR_RC_layer, self).__init__(
            rng=rng, trng=trng, input=input, input_shape=input_shape, filter_figure=filter_figure, n_steps=n_steps,
            pool=pool, pool_mode=pool_mode, L=L, l=l, k_top=k_top, s=s, LRN=LRN,
            W=W, b=b, activation=activation)
        
        self.output = _dropout_from_layer(trng, self.output, p=dropout_rate) 