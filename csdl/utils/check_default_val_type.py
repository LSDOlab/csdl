import numpy as np
from typing import List, Union


def check_default_val_type(
    x: Union[float, int, np.ndarray, List[int], List[float]]
) -> Union[int, float, np.ndarray]:
    y = np.array(x) if isinstance(x, list) else x
    if not isinstance(y, (int, float, np.ndarray)):
        raise TypeError(
            '{} is not an int, float, or Numpy ndarray, or list of floats/ints/ndarrays. Note that sparse matrices are not allowed.'
            .format(x))
    return y
