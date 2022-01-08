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
            raise TypeError(summand, " is not an Variable object")
        if summand.shape != summands[0].shape:
            raise ValueError("Arguments must have the same shape")

    if axes == None:
        if len(summands) == 1:
            shape = (1, )
        else:
            shape = summands[0].shape
    else:
        shape = tuple(np.delete(summands[0].shape, axes))

    op = ops.sum(*summands, axes=axes)
    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    print('SHAPE', op.outs[0].shape)
    print('SHAPE (dep)', op.dependencies[0].shape)
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
