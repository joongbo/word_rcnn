import numpy as np
import theano

from initializers import *

# define a class of a word embedding layer
        
class WELayer(object):
    def __init__(self, rng, V, k, w2v=None):
        ''' 
            [input]
                V: vocabulary size
                k: embedding dimension
            [optional]
                w2v: pretrained word embedding
                
            [output] 
                Words: shared variable for word embedding
        '''
        self.rng = rng
        self.V = V
        self.k = k
        self.w2v = w2v
        
        self.initialize_weights()
        
    def initialize_weights(self, svd_init=False):
        if self.w2v is not None:
            value = self.w2v # pretrained vector
        elif svd_init:
            value = svd_orthonomal(self.rng, [self.V, self.k])
        else:
            value = self.rng.randn(self.V, self.k) / np.sqrt(self.k) # Xavier initialization
        Words = theano.shared(name='Words', 
                              value=value.astype(theano.config.floatX), 
                              borrow=True)
        self.Words = Words
        self.params = [self.Words]