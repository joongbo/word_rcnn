# not implemented

import numpy as np

import theano
from theano import scan
import theano.tensor as T
from theano.tensor.signal import pool

# define pooling methods

def pooling(self):
    ''' spatial pooling '''
    input = self.output
    input_shape = self.output_shape

    s = float(input_shape[2])
    k_l = max(self.k_top, int(s*(self.L-self.l)//self.L))
    pool_shape = (int( s/k_l ), 1)

    self.output = pool.pool_2d(input=input, ws=pool_shape, ignore_border=True)
    self.output_shape = (input_shape[0], input_shape[1], k_l, input_shape[3])

def k_max_pooling(self):
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

def dynamic_k_max_pooling(self):
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