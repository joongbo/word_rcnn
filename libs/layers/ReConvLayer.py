import numpy as np
import theano
from theano import scan
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from activations import *
from initializers import *

from BatchNormalization import *
from LocalResponseNormalization import *

# define a class of a recurrent convolutional layer

class ReConvLayer(object):
    """ Layer of a recurrent convolutional network """
    def __init__(self, rng, trng, input, input_shape, filter_figure, n_steps=1, 
                 LRN=False, BN=False, BN_mode=0,
                 pool=False, pool_mode='max', k_top=1, L=None, l=None, s=None, 
                 W=None, b=None, activation=relu):
        
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
            if pool_mode not in ['k_max', 'd_k_max', 'spatial', 'max']:
                raise TypeError('pool_mode must be one of [k_max, d_k_max, max, spatial], given', pool_mode)
            if L is None or l is None or k_top is None or s is None:
                raise TypeError('any pooling variables must not be None')
                
        # assign class variables
        self.rng = rng
        self.trng = trng
        self.input = T.addbroadcast(input,3)
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

        if W is None and b is None:
            self.initialize_weights()
        else:
            if W is None or b is None:
                raise TypeError('one of parameters is not given', ('W', W, 'b', b)) 
            self.W = W
            self.b = b
            self.params = [self.W, self.b]

        self.LRN = LRN
        if self.LRN:
            self._lrn = LocalResponseNormalization()

        self.BN = BN
        self.BN_mode = BN_mode
        if self.BN:
            self._batch_normalize = BatchNormalization(self.output_shape, mode=self.BN_mode)
            self.params += self._batch_normalize.params
            
        # recurrent convolution output
        def _one_step(out_tm1):
            conv_out = conv2d(input=out_tm1, input_shape=self.input_shape, 
                           filters=self.W, filter_shape=self.filter_shape, 
                           border_mode='half')
            out_t = T.addbroadcast(conv_out,3) + self.b.dimshuffle('x',0,'x','x')
            if self.BN: out_t = self._batch_normalize.get_result(out_t)
            out_t = self.activation(out_t)
            if self.LRN: out_t = self._lrn(out_t)
            #if self.LRN: out_t = self._local_resp_normalize(out_t, self.filter_shape[0])
                    
            return out_t
        
        self.initial = self.input
        out_t, _ = scan(fn=_one_step,
                        outputs_info=dict(initial=self.initial),
                        n_steps=self.n_steps)
        self.output = out_t[-1]
                
        if self.pool:
            if self.pool_mode=='d_k_max':
                self._dynamic_k_max_pooling()
            elif self.pool_mode=='k_max':
                self._k_max_pooling()
            elif self.pool_mode=='max' or self.pool_mode=='spatial':
                self._pooling()
            else:
                raise TypeError('pooling is failed')
            
        
    def initialize_weights(self, svd_init=False):
        if svd_init:
            W_value = svd_orthonomal(self.rng, self.filter_shape)
        else:
            fan_in = np.prod(self.filter_shape[1:])
            fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
            W_value = self.rng.randn(self.filter_shape[0], self.filter_shape[1],
                                     self.filter_shape[2], self.filter_shape[3])
            if self.activation in [relu]:
                W_value = W_value / np.sqrt(fan_in / 2)
            else:
                W_value = W_value / np.sqrt(fan_in)
        
        b_value = np.zeros((self.filter_shape[0],))
        W = theano.shared(name='W', value=W_value.astype(theano.config.floatX), borrow=True)
        b = theano.shared(name='b', value=b_value.astype(theano.config.floatX), borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
    
    # local response normalization
    def _local_resp_normalize(self, input, K, alpha=0.001, beta=0.75):
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
    def _pooling(self):
        input = self.output
        input_shape = self.output_shape
        
        s = float(input_shape[2])
        k_l = max(self.k_top, int(s*(self.L-self.l)//self.L))
        pool_shape = (int( s/k_l ), 1)
        
        self.output = pool.pool_2d(input=input, ws=pool_shape, ignore_border=True)
        self.output_shape = (input_shape[0], input_shape[1], k_l, input_shape[3])

    def _k_max_pooling(self):
        ''' k max pooling '''
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
        
    def _dynamic_k_max_pooling(self):
        ''' dynamic k max pooling '''
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