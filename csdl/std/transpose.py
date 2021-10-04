from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def transpose(var: Variable):
    '''
    This function can perform the transpose of an input

    **Parameters**

    expr: Variable
        The input which will be transposed

    '''
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    op = ops.transpose(var)
    op.outs = (Output(
        None,
        op=op,
        shape=var.shape[::-1],
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
