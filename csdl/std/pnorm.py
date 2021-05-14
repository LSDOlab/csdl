from csdl.comps.vectorized_pnorm_comp import VectorizedPnormComp
from csdl.comps.vectorized_axiswise_pnorm_comp import VectorizedAxisWisePnormComp
from csdl.core.variable import Variable
from typing import List
import numpy as np


def pnorm(expr, pnorm_type=2, axis=None):
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

    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    if axis is not None:
        if not isinstance(axis, int) and not isinstance(axis, tuple):
            raise TypeError("axis must be an integer or tuple of integers")
    out = Variable()
    out.add_dependency_node(expr)

    if pnorm_type % 2 != 0 or pnorm_type <= 0:
        raise Exception(pnorm_type, " is not positive OR is not even")

    else:
        if axis == None:
            out.build = lambda: VectorizedPnormComp(
                shape=expr.shape,
                in_name=expr.name,
                out_name=out.name,
                pnorm_type=pnorm_type,
                val=expr.val,
            )
        else:
            output_shape = np.delete(expr.shape, axis)
            out.shape = tuple(output_shape)

            out.build = lambda: VectorizedAxisWisePnormComp(
                shape=expr.shape,
                in_name=expr.name,
                out_shape=out.shape,
                out_name=out.name,
                pnorm_type=pnorm_type,
                axis=axis if isinstance(axis, tuple) else (axis, ),
                val=expr.val,
            )
    return out
