from csdl.comps.transpose_comp import TransposeComp
from csdl.core.variable import Variable
from typing import List
import numpy as np


def transpose(expr: Variable):
    '''
    This function can perform the transpose of an input

    Parameters
    ----------
    expr: Variable
        The input which will be transposed

    '''
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.add_dependency_node(expr)
    out.shape = expr.shape[::-1]
    out.build = lambda: TransposeComp(
        in_name=expr.name,
        in_shape=expr.shape,
        out_name=out.name,
        out_shape=out.shape,
        val=expr.val,
    )
    return out
