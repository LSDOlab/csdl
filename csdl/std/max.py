from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops

import numpy as np


def max(*vars, axis=None, rho=20.):
    '''
    This function can compute an elementwise or axiswise minimum of
    a single or multiple inputs.

    **Parameters**

    vars: Variable(s)
        The Variable(s) over which to take the minimum

    axis: int
        The axis along which the minimum will be taken over

    rho: float
        This is a smoothing parameter, which dictates how smooth or sharp
        the minimum is
    '''

    for v in vars:
        if not isinstance(v, Variable):
            raise TypeError(v, " is not a Variable object")
        if v.shape != vars[0].shape:
            raise ValueError("Arguments must have the same shape")
        l = len(vars[0].shape)
    if axis is not None:
        if axis > l:
            raise ValueError(
                "axis {} is bounds for array of dimension {}".format(
                    axis, l))

        # finished error checking
        if len(vars) == 1:
            # max along axis of a single array
            if len(vars[0].shape) > 1:
                # tuple guaranteed to be nonempty when taking max along
                # axis of a matrix or tensor
                shape = tuple(np.delete(vars[0].shape, (axis, )))
            else:
                # in this cae, we can't simply delete an axis because
                # that would result in an empty tuple
                shape = (1, )
        else:
            # tuple guaranteed to be nonempty when summing over axes
            # of matrices and tensors
            if len(vars[0].shape) > 1:
                shape = tuple(np.delete(vars[0].shape, (axis, )))
            else:
                shape = (1, )
    else:
        # axes == None
        if len(vars) == 1:
            # take max of all elements of a single array, regardless of
            # shape
            shape = (1, )
        else:
            # take max of each set of elements with same array indices
            # across all arrays, result is same shape
            shape = vars[0].shape

    op = ops.max(*vars, rho=rho, axis=axis)
    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
