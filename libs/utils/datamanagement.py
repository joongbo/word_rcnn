import cPickle as pkl
import numpy as np


def load_w2v(data_file):
    pkl_f = open(data_file, 'rb')
    word2vec, word2idx = pkl.load(pkl_f)
    pkl_f.close()
    
    return word2vec, word2idx

# load twitter dataset from .pkl file
def load_sst(data_file):
    pkl_f = open(data_file, 'rb')
    train, valid, test, vocab = pkl.load(pkl_f)
    pkl_f.close()
    
    # split training set into validation set
    data_x, data_y, data_l = train
    if len(data_x) != len(data_y) or len(data_y) != len(data_l):
        raise IOError('Bad input: lengths are not matched')
    
    data = [train, valid, test]
    return data, vocab

# load twitter dataset from .pkl file
def load_s140(data_file, num_valid=12000):
    pkl_f = open(data_file, 'rb')
    train, test, vocab = pkl.load(pkl_f)
    pkl_f.close()
    
    # split training set into validation set
    data_x, data_y, data_l = train
    if len(data_x) != len(data_y) or len(data_y) != len(data_l):
        raise IOError('Bad input: lengths are not matched')
    num_samples = len(data_y)
    sidx = np.random.permutation(num_samples)
    
    valid_x = []
    valid_y = []
    valid_l = []
    train_x = []
    train_y = []
    train_l = []
    for i in xrange(num_samples):
        if i < num_valid:
            valid_x.append(data_x[ sidx[i] ])
            valid_y.append(data_y[ sidx[i] ])
            valid_l.append(data_l[ sidx[i] ])
        else:
            train_x.append(data_x[ sidx[i] ])
            train_y.append(data_y[ sidx[i] ])
            train_l.append(data_l[ sidx[i] ])

    train = [train_x, train_y, train_l]
    valid = [valid_x, valid_y, valid_l]

    data = [train, valid, test]
    return data, vocab


# fit length of each data
def fit_length(data, max_len):
    data_x, data_y, data_l = data
    
    list_zeros = np.zeros(max_len, 'int32').tolist()
    fl_data_x = []
    fl_data_l = []
    for datum_x, datum_l in zip(data_x, data_l):
        if datum_l >= max_len:
            fl_data_x.append( datum_x[:max_len] )
            fl_data_l.append( max_len )
        else:
            fl_data_x.append( datum_x + list_zeros[:(max_len-datum_l)] )
            fl_data_l.append( datum_l )
    
    np_data_x = np.asarray(fl_data_x, dtype='int32')
    np_data_y = np.asarray(data_y, dtype='int32')
    np_data_l = np.asarray(fl_data_l, dtype='int32')
    
    data = [np_data_x, np_data_y, np_data_l]
    return data


# fit batch size
def fit_batch_add(data, mb_size):
    data_x, data_y, data_l = data
    data_size = len(data_y)
    if data_size % mb_size > 0:
        extra_n = mb_size - data_size % mb_size
        idxs = np.random.choice(a=data_size, size=extra_n, replace=False)
        data_x = np.concatenate([data_x, data_x[idxs]])
        data_y = np.concatenate([data_y, data_y[idxs]])
        data_l = np.concatenate([data_l, data_l[idxs]])

    data = [data_x, data_y, data_l]
    return data

def fit_batch_del(data, mb_size):
    data_x, data_y, data_l = data
    data_size = len(data_y)
    if data_size % mb_size > 0:
        extra_n = data_size % mb_size
        idxs = np.random.choice(a=data_size, size=extra_n, replace=False)
        data_x = np.delete(data_x, idxs, 0)
        data_y = np.delete(data_y, idxs, 0)
        data_l = np.delete(data_l, idxs, 0)

    data = [data_x, data_y, data_l]
    return data


# shuffle training data at each epoch
def shuffle_data(data):
    data_x, data_y, data_l= data
    ridx = np.random.permutation( len(data_y) )
    data_x = data_x[ridx]
    data_y = data_y[ridx]
    data_l = data_l[ridx]

    data = [data_x, data_y, data_l]
    return data