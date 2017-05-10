import numpy as np
import theano
from theano import scan
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from activations import *
from initializers import *

# define a class of a recurrent convolutional layer

class RCLayer(object):
    """ Layer of a recurrent convolutional network """
    def __init__(self, rng, trng, input, input_shape, filter_figure, n_steps, 
                 LRN=False, pool=False, pool_mode='max', k_top=1, L=None, l=None, s=None, 
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
            self.initialize_weights()
        else:
            if W is None or b is None:
                raise TypeError('one of parameters is not given', ('W', W, 'b', b)) 
            self.W = W
            self.b = b
            self.params = [self.W, self.b]
        
        
        # recurrent convolution output
        def _one_step(out_tm1):
            conv_out = conv2d(input=out_tm1, input_shape=self.input_shape, 
                           filters=self.W, filter_shape=self.filter_shape, 
                           border_mode='half')
            out_t = self.input + conv_out + self.b.dimshuffle('x',0,'x','x')
            out_t = self.activation(out_t)
            if self.LRN: out_t = self._local_resp_normalize(out_t, self.filter_shape[0])
                    
            return out_t
        
        self.initial = T.unbroadcast(self.input, 3)
        out_t, _ = scan(fn=_one_step,
                        outputs_info=dict(initial=self.initial),
                        n_steps=self.n_steps)
        self.output = out_t[-1]
                
        if self.pool:
            if self.pool_mode is 'd_k_max':
                self._dynamic_kmax_pooling()
            elif self.pool_mode=='k_max':
                self._kmax_pooling()
            elif self.pool_mode=='max' or self.pool_mode=='spatial':
                self._pooling()
        
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
        
    # response normalization
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

    def _kmax_pooling(self):
        raise NotImplementedError()
    def _dynamic_kmax_pooling(self):
        raise NotImplementedError()