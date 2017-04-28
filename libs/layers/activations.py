import theano.tensor as T

# define activation functions

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