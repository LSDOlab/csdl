from csdl.comps.axiswise_min_comp import AxisMinComp
from csdl.comps.elementwise_min_comp import ElementwiseMinComp
from csdl.comps.scalar_extremum_comp import ScalarExtremumComp

from csdl.core.variable import Variable

import numpy as np


def min(*exprs, axis=None, rho=20.):
    '''
    This function can compute an elementwise or axiswise minimum of
    a single or multiple inputs.

    Parameters
    ----------
    exprs: Variable(s)
        The Variable(s) over which to take the minimum

    axis: int
        The axis along which the minimum will be taken over

    rho: float
        This is a smoothing parameter, which dictates how smooth or sharp
        the minimum is
    '''

    out = Variable()
    for expr in exprs:
        if not isinstance(expr, Variable):
            raise TypeError(expr, " is not an Variable object")
        out.add_dependency_node(expr)

    if len(exprs) == 1 and axis != None:
        output_shape = np.delete(expr.shape, axis)
        out.shape = tuple(output_shape)

        out.build = lambda: AxisMinComp(
            shape=exprs[0].shape,
            in_name=exprs[0].name,
            axis=axis,
            out_name=out.name,
            rho=rho,
            val=exprs[0].val,
        )

    elif len(exprs) > 1 and axis == None:

        shape = exprs[0].shape
        for expr in exprs:
            if shape != expr.shape:
                raise Exception("The shapes of the inputs must match!")

        out.shape = expr.shape

        out.build = lambda: ElementwiseMinComp(
            shape=expr.shape,
            in_names=[expr.name for expr in exprs],
            out_name=out.name,
            rho=rho,
            vals=[expr.val for expr in exprs],
        )

    elif len(exprs) == 1 and axis == None:

        out.build = lambda: ScalarExtremumComp(
            shape=exprs[0].shape,
            in_name=exprs[0].name,
            out_name=out.name,
            rho=rho,
            lower_flag=True,
            val=exprs[0].val,
        )

    else:
        raise Exception("Do not give multiple inputs and an axis")
    return out
