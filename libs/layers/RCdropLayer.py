from activations import *
from dropout import *
from RCLayer import RCLayer

# define a class of a recurrent convolutional layer with dropout

class RCdropLayer(RCLayer):
    ''' Layer of a recurrent convolutional network with dropout '''
    def __init__(self, rng, trng, input, input_shape, filter_figure, n_steps, 
                 LRN=False, pool=False, pool_mode='max', k_top=None, L=None, l=None, s=None, 
                 p=0.5, W=None, b=None, activation=tanh):
        super(RCdropLayer, self).__init__(
            rng=rng, trng=trng, input=input, input_shape=input_shape, filter_figure=filter_figure, n_steps=n_steps,
            LRN=LRN, pool=pool, pool_mode=pool_mode, k_top=k_top, L=L, l=l, s=s, 
            W=W, b=b, activation=activation)
        
        self.p = p
        self.output = dropout_from_layer(self.trng, self.output, p=p)