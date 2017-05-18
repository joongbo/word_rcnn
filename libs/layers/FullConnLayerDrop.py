from activations import *
from dropout import *
from FullConnLayer import FullConnLayer

# define a class of a convolutional layer with dropout

class FullConnLayerDrop(FullConnLayer):
    ''' Layer of a fully connected network with dropout '''
    def __init__(self, rng, trng, input, n_in, n_out, BN=False, BN_mode=0,
                 p=0.5, W=None, b=None, activation=tanh):
        super(FullConnLayerDrop, self).__init__(
            rng=rng, trng=trng, input=input, n_in=n_in, n_out=n_out, BN=BN, BN_mode=BN_mode,
            W=W, b=b, activation=activation)
        
        self.p = p
        self.output = dropout_from_layer(self.trng, self.output, p=self.p)