import numpy as np

# define a function of svd orthonormal weight initialization
        
def svd_orthonomal(rng, shape):
    ''' svd orthonormal '''
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = rng.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    valueW = q.reshape(shape)
    
    return valueW