import numpy as np
import theano
from theano import scan
import theano.tensor as T

from activations import *
from initializers import *

from BatchNormalization import *

# define a class of a fully connected layer

class FullConnLayer(object):
    """ Layer of a fully connected (or feed forward) network """
    def __init__(self, rng, trng, input, n_in, n_out, BN=False, BN_mode=0,
                 W=None, b=None, activation=tanh):
        
        self.rng = rng
        self.trng = trng
        self.input = input
        self.mini_batch = input.shape[0]
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if W is None and b is None:
            self.initialize_weights()
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
        
        self.BN = BN
        self.BN_mode = BN_mode
        if self.BN:
            self._batch_normalize = BatchNormalization(self.output_shape, mode=self.BN_mode)
            self.params += self._batch_normalize.params
            self.output = self._batch_normalize.get_result(self.output)

        self.output = self.activation(self.output)
        
        
    def initialize_weights(self, svd_init=False):
        if svd_init:
            W_value = svd_orthonomal(self.rng, [self.n_in, self.n_out])
        else:
            if self.activation in [relu]:
                W_value = self.rng.randn(self.n_in, self.n_out) / np.sqrt(self.n_in/2)
            else:
                W_value = self.rng.randn(self.n_in, self.n_out) / np.sqrt(self.n_in)
        b_value = np.zeros((self.n_out,))
        W = theano.shared(name='W', value=W_value.astype(theano.config.floatX), borrow=True)
        b = theano.shared(name='b', value=b_value.astype(theano.config.floatX), borrow=True)
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