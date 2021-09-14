from typing import Tuple, Union
import numpy as np
from numbers import Number


def get_shape_val(
    shape: Union[int, Tuple[int]],
    val: Union[int, float, np.ndarray],
) -> Tuple[Tuple[int], np.ndarray]:
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
        # if isinstance(val, Number):
        return shape, val * np.ones(shape)
    if isinstance(val, list):
        val = np.array(val)
    if isinstance(val, np.ndarray):
        if shape == (1, ):
            return val.shape, val
        if val.shape != shape:
            raise ValueError("Value shape mismatch")
    return shape, val
