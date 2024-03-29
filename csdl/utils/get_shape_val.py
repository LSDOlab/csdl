from typing import Tuple, List, Union
import numpy as np


def get_shape_val(
    shape: Union[int, Tuple[int]],
    val: Union[int , float , np.ndarray , List[int] ,List[float]],
    no_val:bool = False, # if True, do not calculate value
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """
    Get shape from shape or value if shape is unspecified

    **Parameters**

    shape: None or tuple
        Shape of value

    val: Number or ndarray
        Value

    **Returns**

    Tuple[int]
        Shape of value
    """
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(val, (int, float)):
        if no_val:
            val = None
        else:
            val = val * np.ones(shape)
    if isinstance(val, list):
        if no_val:
            val = None
        else:
            val = np.array(val)
    if isinstance(val, np.ndarray):
        if shape == (1, ):
            return val.shape, val
        if not no_val:
            if val.shape != shape:
                raise ValueError(
                    "Value shape mismatch. val has shape {}, and shape is {}"
                    .format(val.shape, shape))
    return shape, val
