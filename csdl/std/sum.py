from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def sum(*summands: Variable, axes=None):
    '''
    This function can compute an elementwise or axiswise sum of
    a single or multiple inputs.

    **Parameters**

    summands: Variable(s)
        The Variable(s) over which to take the sum

    axes: tuple[int]
        The axes along which the sum will be taken over

    '''
    for summand in summands:
        if not isinstance(summand, Variable):
            raise TypeError(summand, " is not a Variable object")
        if summand.shape != summands[0].shape:
            raise ValueError("Arguments must have the same shape")
        l = len(summands[0].shape)
        m = max(summands[0].shape)
    if isinstance(axes, tuple):
        if len(axes) > l:
            raise ValueError("To many axes specified")
        for ax in axes:
            if ax > m:
                raise ValueError(
                    "axes {} and greater in axes {} are out of bounds for array of dimension {}"
                    .format(ax, axes, l))

        # finished error checking
        if len(summands) == 1:
            # sum over axes of a single array
            if len(summands[0].shape) == 1:
                # in this cae, we can't simply delete an axis because
                # that would result in an empty tuple
                shape = (1, )
            else:
                # tuple guaranteed to be nonempty when summing over axes
                # of matrices and tensors
                if len(summands[0].shape) > len(axes):
                    shape = tuple(np.delete(summands[0].shape, axes))
                else:
                    shape = (1, )
        else:
            # tuple guaranteed to be nonempty when summing over axes
            # of matrices and tensors
            if len(summands[0].shape) > len(axes):
                shape = tuple(np.delete(summands[0].shape, axes))
            else:
                shape = (1, )
    else:
        # axes == None
        if len(summands) == 1:
            # sum all elements of a single array, regardless of shape
            shape = (1, )
        else:
            # sum arrays together, result is same shape
            shape = summands[0].shape

    op = ops.sum(*summands, axes=axes)
    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
