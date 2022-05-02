from csdl.lang.variable import Variable
from csdl.lang.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def outer(a: Variable, b: Variable):
    '''
    This can the outer product between two inputs.

    **Parameters**

    expr1: Variable
        The first input for the outer product.

    expr2: Variable
        The second input for the outer product.

    '''
    if not isinstance(a, Variable):
        raise TypeError(a, " is not an Variable object")
    elif not isinstance(b, Variable):
        raise TypeError(b, " is not an Variable object")

    shape = tuple(list(a.shape) + list(b.shape))
    op = ops.outer(a, b)
    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
