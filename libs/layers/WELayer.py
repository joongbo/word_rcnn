import numpy as np
import theano

from initializers import *

# define a class of a word embedding layer
        
class WELayer(object):
    def __init__(self, rng, V, k):
        ''' 
            [input]
                V: vocabulary size
                k: embedding dimension
            [output] 
                Words: shared variable for word embedding
        '''
        self.rng = rng
        self.V = V
        self.k = k
        
        self.initialize_weights()
        
    def initialize_weights(self):
        ''' Xavier Initialization '''
        value = self.rng.randn(self.V, self.k) / np.sqrt(self.k)
        Words = theano.shared(name='Words', 
                              value=value.astype(theano.config.floatX), 
                              borrow=True)
        self.Words = Words
        self.params = [self.Words]
        
    def svd_initialize_weights(self):
        value = svd_orthonomal(self.rng, [self.V, self.k])
        Words = theano.shared(name='Words', 
                              value=value.astype(theano.config.floatX), 
                              borrow=True)
        self.Words = Words
        self.params = [self.Words]