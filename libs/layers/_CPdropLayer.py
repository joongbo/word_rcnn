from activations import *
from dropout import *
from CPLayer import CPLayer

# define a class of a convolutional layer with dropout

class CPdropLayer(CPLayer):
    ''' layer of a convolutional network with dropout '''
    def __init__(self, rng, trng, input, input_shape, filter_figure, border_mode,
                 LRN=False, pool=False, pool_mode='max', k_top=None, L=None, l=None, s=None, 
                 p=0.5, W=None, b=None, activation=tanh):
        super(CPdropLayer, self).__init__(
            rng=rng, trng=trng, input=input, input_shape=input_shape, filter_figure=filter_figure, border_mode=border_mode,
            LRN=LRN, pool=pool, pool_mode=pool_mode, k_top=k_top, L=L, l=l, s=s, 
            W=W, b=b, activation=activation)
        
        self.p = p
        self.output = dropout_from_layer(self.trng, self.output, p=p)