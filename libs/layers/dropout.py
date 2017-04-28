import theano
import theano.tensor as T

# define dropout

def dropout_from_layer(trng, input, p):
    ''' dropout '''
    mask = T.cast(trng.binomial(n=1, p=1-p, size=input.shape), theano.config.floatX)
    output = input * mask
    return output