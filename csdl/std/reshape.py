from csdl.lang.variable import Variable
from csdl.lang.output import Output
import csdl.operations as ops
import numpy as np


def reshape(var: Variable, new_shape: tuple):
    '''
    This function reshapes the input into a new shape.

    **Parameters**

    var: Variable
        The Variable which you want to reshape

    new_shape: tuple[int]
        A tuple of ints specifying the new shape desired
    '''
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    if np.prod(var.shape) != np.prod(new_shape):
        raise ValueError(
            "Cannot reshape variable of shape {} into shape {}".format(
                var.shape, new_shape))
    op = ops.reshape(var)
    op.outs = (Output(
        None,
        op=op,
        shape=new_shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]


def flatten(var: Variable):
    '''
    This function flattens the input into a 1D array.

    **Parameters**

    var: Variable
        The Variable which you want to flatten
    '''
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    return reshape(var, (var.size,))