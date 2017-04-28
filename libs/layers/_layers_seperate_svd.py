import numpy as np

import theano
from theano import scan
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, batch_normalization


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

def svd_orthonormal(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q

""" Classes """
class batch_normalize(object) :
    def __init__(self, input_shape, mode=0, momentum=0.9):
        self.ndim = len(input_shape)
        self.input_shape = input_shape
        self.momentum = momentum
        self.mode = 0 # mode : 0 means training, 1 means inference
    
        # random setting of gamma and beta, setting initial mean and std
        self.gamma = theano.shared(np.ones((self.input_shape[1]), dtype=theano.config.floatX), name='gamma', borrow=True)
        self.beta = theano.shared(np.zeros((self.input_shape[1]), dtype=theano.config.floatX), name='beta', borrow=True)
        self.mean = theano.shared(np.zeros((self.input_shape[1]), dtype=theano.config.floatX), name='mean', borrow=True)
        self.var = theano.shared(np.ones((self.input_shape[1]), dtype=theano.config.floatX), name='var', borrow=True)

        # parameter save for update
        self.params = [self.gamma, self.beta]

        
    def get_result(self, input):
        # returns BN result for given input.
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
            # in CNN mode, gamma and beta exists for every single channel separately.
            # for each channel, calculate mean and std for (mini_batch_size * row * column) elements.
            # then, each channel has own scalar gamma/beta parameters.
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
        repeated = T.repeat(vec, self.input_shape[2]*self.input_shape[3])
        return repeated.reshape((self.input_shape[1],self.input_shape[2],self.input_shape[3]))
    
    
""" Layers """
class FC_layer(object):
    """ Layer of a fully connected (or feed forward) network """
    def __init__(self, rng, trng, input, n_in, n_out, 
                 BN=False, BN_mode=0, Drop=False,
                 W=None, b=None, activation=tanh):

        self.input = input
        self.mini_batch = input.shape[0]
        self.n_in = n_in
        self.n_out = n_out
        self.BN = BN
        self.Drop = Drop
        self.activation = activation
        
        if W is None:
#            bound = np.sqrt(6. / (self.n_in + self.n_out))
#            value = rng.uniform(low=-bound, high=bound, size=(self.n_in, self.n_out))
            value = svd_orthonormal((self.n_in, self.n_out))
            W = theano.shared(name='W', value=value.astype(theano.config.floatX), borrow=True)
        if b is None:
            value = np.zeros((self.n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=value, name='b', borrow=True)
        self.W = W
        self.b = b

         # parameters of the model
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b
        self.output_shape = (input.shape[0], n_out)
        
        if self.Drop:
            self.drop_out(trng)
            
        if self.BN:
            self.BN_layer = batch_normalize(self.output_shape, mode=BN_mode)
            self.BN_params = self.BN_layer.params
            self.output = self.BN_layer.get_result(self.output)
        self.output = self.activation(self.output)
        
    # drop out function
    def drop_out(self, trng, p=0.5, rescale=True):
        one = T.constant(1)
        retain_p = one - p
        if rescale:
            self.output /= retain_p
        mask = trng.binomial(size=self.output.shape, p=retain_p, dtype=self.output.dtype)
        
        self.output = self.output * mask
            
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
                 pool=False, pool_mode='k_max', L=None, l=None, k_top=None, s=None, 
                 BN=False, BN_mode=0, RN=False, LRN=False, Drop=False, 
                 W=None, b=None, activation=relu):
        
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

        self.BN = BN
        self.RN = RN
        self.LRN = LRN
        self.Drop = Drop
        
        if W is None:
#             fan_in = np.prod(self.filter_shape[1:])
#             fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
#             bound = np.sqrt(6. / (fan_in + fan_out))
#             value = rng.uniform(low=-bound, high=bound, size=self.filter_shape)
            value = svd_orthonormal(filter_shape)
            W =  theano.shared(value=value.astype(dtype=theano.config.floatX), name='W', borrow=True)
        if b is None:
            value = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=value, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        
        if self.BN:
            self.BN_layer = batch_normalize(self.output_shape, mode=BN_mode)
            self.BN_params = self.BN_layer.params
            
        # convolution output (feed forward)
        conv_out = conv2d(input=self.input, input_shape=self.input_shape, 
                          filters=self.W, filter_shape=self.filter_shape, 
                          border_mode=self.border_mode)
        self.output = conv_out + self.b.dimshuffle('x',0,'x','x')
        
        if self.BN: self.output = self.BN_layer.get_result(self.output)
        self.output = self.activation(self.output) # activation
        
        if self.RN: self.response_normalize()
        elif self.LRN: self.local_resp_normalize(self.filter_shape[0])
        if self.Drop: self.drop_out(trng)
        if self.pool:
            if self.pool_mode is 'd_k_max':
                self.dynamic_k_max_pooling()
            elif self.pool_mode=='k_max':
                self.k_max_pooling()
            elif self.pool_mode=='max' or self.pool_mode=='spatial':
                self.pooling()
        
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
        
    def response_normalize(self, alpha=0.001, beta=0.75):
        squared = self.output ** 2
        normalizer = (1. + alpha*( T.mean(squared, axis=1) ) )**beta
        
        self.output = self.output / normalizer.dimshuffle(0,'x',1,2)
        
    def drop_out(self, trng, p=0.5, rescale=True):
        one = T.constant(1)
        retain_p = one - p
        if rescale:
            self.output /= retain_p
        mask = trng.binomial(size=self.output.shape, p=retain_p, dtype=self.output.dtype)
        
        self.output = self.output * mask
        
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
                 pool=True, pool_mode='k_max', L=None, l=None, k_top=None, s=None, 
                 BN=False, BN_mode=0, RN=False, LRN=False, Drop=False, 
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

        self.BN = BN
        self.RN = RN
        self.LRN = LRN
        self.Drop = Drop
        
        if W is None:
#             fan_in = np.prod(self.filter_shape[1:])
#             fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])
#             bound = np.sqrt(6. / (fan_in + fan_out))
#             value = rng.uniform(low=-bound, high=bound, size=self.filter_shape)
            value = svd_orthonormal(filter_shape)
            W =  theano.shared(value=value.astype(dtype=theano.config.floatX), name='W', borrow=True)
        if b is None:
            value = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=value, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        
        if self.BN:
            self.BN_layer = batch_normalize(self.output_shape, mode=BN_mode)
            self.BN_params = self.BN_layer.params
        
        # recurrent convolution output
        def one_step(out_tm1):
            conv_out = conv2d(input=out_tm1, input_shape=self.input_shape, 
                           filters=self.W, filter_shape=self.filter_shape, 
                           border_mode='half')
            out_t = self.input + conv_out + self.b.dimshuffle('x',0,'x','x')
            if self.BN: out_t = self.BN_layer.get_result(out_t)
            out_t = self.activation(out_t)
            if self.RN: out_t = self.response_normalize(out_t)
            elif self.LRN: out_t = self.local_resp_normalize(out_t, self.filter_shape[0])
                    
            return out_t
        
        self.initial = T.unbroadcast(self.input, 3)
        out_t, _ = scan(fn=one_step,
                        outputs_info=dict(initial=self.initial),
                        n_steps=self.n_steps)
        self.output = out_t[-1]
        
        if self.Drop: self.drop_out(trng)
        
        if self.pool:
            if self.pool_mode is 'd_k_max':
                self.dynamic_k_max_pooling()
            elif self.pool_mode=='k_max':
                self.k_max_pooling()
            elif self.pool_mode=='max' or self.pool_mode=='spatial':
                self.pooling()
        
                                      
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
        
    def response_normalize(self, input, alpha=0.001, beta=0.75):
        squared = input ** 2
        normalizer = (1. + alpha*( T.mean(squared, axis=1) ) )**beta
        
        return input / normalizer.dimshuffle(0,'x',1,2)
        
    def drop_out(self, trng, p=0.5, rescale=True):
        one = T.constant(1)
        retain_p = one - p
        if rescale:
            self.output /= retain_p
        mask = trng.binomial(size=self.output.shape, p=retain_p, dtype=self.output.dtype)
        
        self.output = self.output * mask
        
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