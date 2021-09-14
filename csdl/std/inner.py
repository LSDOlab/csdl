from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def inner(a: Variable, b: Variable, axes=None):
    '''
    This can the dot product between two inputs.

    **Parameters**

    a: Variable
        The first input for the inner product.

    b: Variable
        The second input for the inner product.

    axes: tuple[int]
        The axes along which the inner product is taken.
    '''
    if not isinstance(a, Variable):
        raise TypeError(a, " is not an Variable object")
    elif not isinstance(b, Variable):
        raise TypeError(b, " is not an Variable object")

    # compute shape for output
    if len(a.shape) == 1 and len(b.shape) == 1:
        if axes is not None:
            raise ValueError(
                "Axes must not be provided when both arguments are vectors"
            )
        shape = (1, )
    else:
        if axes is None:
            raise ValueError(
                "Axes must be provided when either argument is a matrix or tensor"
            )
        if len(axes) != 2:
            raise ValueError(
                "One set of axes per argument must be provided when either argument is a matrix or tensor"
            )
        new_in0_shape = np.delete(list(a.shape), axes[0])
        new_in1_shape = np.delete(list(b.shape), axes[1])
        shape = tuple(np.append(new_in0_shape, new_in1_shape))

    op = ops.inner(a, b, axes=axes)
    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    for out in op.outs:
        out.add_dependency_node(op)

    return op.outs[0]
