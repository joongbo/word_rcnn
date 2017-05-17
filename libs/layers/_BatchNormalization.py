import theano
import theano.tensor as T
import numpy as np
import math
import time
from theano.tensor.nnet import conv2d, batch_normalization

class BatchNormalization(object) :
    def __init__(self, input_shape, mode=0 , momentum=0.9) :
        self.ndim = len(input_shape)
        self.input_shape = input_shape
        self.momentum = momentum
        self.mode = mode # mode : 0 means training, 1 means inference
        
        self.initialize_weights()
                
    def initialize_weights(self, svd_init=False):
        rng = np.random.RandomState(int(time.time()))
        # random setting of gamma and beta, setting initial mean and std
        self.gamma = theano.shared(np.ones((self.input_shape[1]), dtype=theano.config.floatX), name='gamma', borrow=True)
        self.beta = theano.shared(np.zeros((self.input_shape[1]), dtype=theano.config.floatX), name='beta', borrow=True)
        self.mean = theano.shared(np.zeros((self.input_shape[1]), dtype=theano.config.floatX), name='mean', borrow=True)
        self.var = theano.shared(np.ones((self.input_shape[1]), dtype=theano.config.floatX), name='var', borrow=True)
        # parameter save for update
        self.params = [self.gamma, self.beta]

    def get_result(self, input):
        epsilon = 1e-06

        if self.ndim==2 :
            if self.mode==0:
                now_mean = T.mean(input, axis=0)
                now_var = T.var(input, axis=0)
                now_std = T.sqrt(now_var+epsilon)
                output = batch_normalization(input, gamma=self.gamma, beta=self.beta, 
                                             mean=now_mean, std=now_std, mode='high_mem') 
                # mean, var update
                self.mean = self.momentum * self.mean + (1.0-self.momentum) * now_mean
                self.var = self.momentum * self.var + (1.0-self.momentum) * \
                    (1.0*self.input_shape[0]/(self.input_shape[0]-1)*now_var)
            else:
                now_std = T.sqrt(self.var+epsilon)
                output = batch_normalization(input, gamma=self.gamma, beta=self.beta, 
                                             mean=self.mean, std=now_std, mode='high_mem')
        else: 
            if self.mode==0 :
                now_mean = T.mean(input, axis=(0,2,3))
                now_var = T.var(input, axis=(0,2,3))
                # mean, var update
                self.mean = self.momentum * self.mean + (1.0-self.momentum) * now_mean
                self.var = self.momentum * self.var + (1.0-self.momentum) * \
                    (1.0*self.input_shape[0]/(self.input_shape[0]-1)*now_var)
            else :
                now_mean = self.mean
                now_var = self.var
            # change shape to fit input shape
            now_mean = self.change_shape(now_mean)
            now_std = self.change_shape(T.sqrt(now_var+epsilon))
            now_gamma = self.change_shape(self.gamma)
            now_beta = self.change_shape(self.beta)

            output = batch_normalization(input, gamma=now_gamma, beta=now_beta, 
                                         mean=now_mean, std=now_std, mode='high_mem')

        return output

    # changing shape for CNN mode
    def change_shape(self, vec) :
        return T.repeat(vec, self.input_shape[2]*self.input_shape[3]).reshape((self.input_shape[1],self.input_shape[2],self.input_shape[3]))
