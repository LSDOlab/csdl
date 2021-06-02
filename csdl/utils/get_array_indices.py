import numpy as np


def get_array_indices(*shape):
    return np.arange(np.prod(shape)).reshape(shape)