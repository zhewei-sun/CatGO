import numpy as np

ep = 0.00001

def log_likelihood(p):
    return -np.sum(np.log(p + ep))

def normalize(array, axis=1):
    denoms = np.sum(array, axis=axis) + ep
    if axis == 1:
        return array / denoms[:,np.newaxis]
    if axis == 0:
        return array / denoms[np.newaxis, :]

