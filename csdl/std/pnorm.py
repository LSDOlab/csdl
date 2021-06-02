from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def pnorm(var, pnorm_type=2, axis=None):
    '''
    This function computes the pnorm

    Parameters
    ----------
    expr: Variable
        The Variable(s) over which to take the minimum

    pnorm_type: int
        This specifies what pnorm to compute. Values must be nonzero positive and even.

    axis: int
        Specifies the axis over which to take the pnorm
    '''

    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    if pnorm_type % 2 != 0 or pnorm_type <= 0:
        raise ValueError(
            "pnorm_type {} is not positive and even".format(pnorm_type))
    if axis is not None:
        if not isinstance(axis, int) and not isinstance(axis, tuple):
            raise ValueError("axis must be an integer or tuple of integers")
        if isinstance(axis, int):
            axis = (axis, )

    op = ops.pnorm(var, pnorm_type=pnorm_type, axis=axis)
    op.outs = (Output(
        None,
        op=op,
        shape=op.dependencies[0].shape,
    ), )
    for out in op.outs:
        out.add_dependency_node(op)

    return op.outs[0]
