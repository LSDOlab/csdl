from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops

import numpy as np


def min(*vars, axis=None, rho=20.):
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

    for var in vars:
        if not isinstance(var, Variable):
            raise TypeError(var, " is not an Variable object")

    if len(vars) == 1 and axis is not None:
        shape = tuple(np.delete(var.shape, axis))
    elif len(vars) == 1 and axis == None:
        shape = vars[0].shape
    if len(vars) > 1 and axis is not None:
        raise RuntimeError(
            "Cannot take minimum of multiple inputs when axis is provided"
        )
    if len(vars) > 1 and axis is None:
        shape = vars[0].shape
        for var in vars:
            if shape != var.shape:
                raise ValueError("The shapes of the inputs must match")

    op = ops.min(*vars, rho=rho, axis=axis)
    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    for out in op.outs:
        out.add_dependency_node(op)

    return op.outs[0]
