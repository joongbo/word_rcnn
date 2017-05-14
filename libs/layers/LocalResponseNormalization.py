from theano.compat.six.moves import xrange
import theano.tensor as T

class LocalResponseNormalization(object):
    def __init__(self, alpha = 1e-4, beta=0.75, n=41):
        self.__dict__.update(locals())
        del self.self
        if n%2==0:
            raise NotImplementedError("Only works with odd n for now")

    def __call__(self, input):
        half = self.n // 2
        sq = T.sqr(input)
        b, ch, r, c = input.shape
        extra_channels = T.alloc(0., b, ch + 2*half, r, c)
        sq = T.set_subtensor(extra_channels[:,half:half+ch,:,:], sq)
        scale = 1

        for i in xrange(self.n):
            scale += self.alpha * sq[:,i:i+ch,:,:]

        scale = scale ** self.beta

        return input / scale