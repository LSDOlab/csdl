from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def dot(a: Variable, b: Variable, axis=None):
    '''
    This can the dot product between two inputs.

    **Parameters**

    expr1: Variable
        The first input for the dot product.

    expr2: Variable
        The second input for the dot product.

    axis: int
        The axis along which the dot product is taken. The axis must
        have an axis of 3.
    '''
    if not (isinstance(a, Variable) and isinstance(b, Variable)):
        raise TypeError("Arguments must both be Variable objects")
    if a.shape != b.shape:
        raise ValueError("The shapes of the inputs must match!")

    op = ops.dot(a, b, axis=axis)
    if len(a.shape) == 1:
        shape = (1, )
    else:
        if axis is None:
            raise ValueError(
                "Axis required when first argument is a matrix or tensor"
            )
        if a.shape[axis] != 3:
            raise ValueError(
                "The specified axis must correspond to the value of 3 in shape"
            )
        shape = tuple(np.delete(list(a.shape), axis))

    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
