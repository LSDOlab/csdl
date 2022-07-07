from csdl.lang.variable import Variable
from csdl.lang.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def average(*operands: List[Variable], axes=None):
    '''
    This function can compute the average of a single input, multiple inputs, or
    along an axis.

    **Parameters**

    operands: Variables
        The Variable(s) over which to take the average


    axes: tuple[int]
        Axes along which to take the average, default value is None

    '''
    for expr in operands:
        if not isinstance(expr, Variable):
            raise TypeError(expr, " is not an Variable object")

    if axes == None:
        shape = operands[0].shape
    else:
        shape = tuple(np.delete(operands[0].shape, axes))

    op = ops.average(*operands, axes=axes)
    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
