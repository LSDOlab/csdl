import csdl.operations as ops
from csdl.lang.variable import Variable
from csdl.lang.output import Output
import numpy as np
from numbers import Number

def exp_a(a:(Variable, np.ndarray, Number), var:Variable):

    """
    computes a^var

    parameters:
    ==========
    a: Variable or array or Number
    var: Variable
    """

    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")

    # if a is not a scalar,
    # a shape must match var shape
    if isinstance(a, (np.ndarray, Variable)):
        a_shape = a.shape
        if (a_shape != (1,)):
            if a_shape != var.shape:
                raise ValueError(f'Constant \'a\' must be a scalar or have same shape as \'{var.name}\'. {a_shape} != {var.shape}.')
    elif not isinstance(a, Number):
        raise TypeError(f'\'a\' must be a csdl variable, array or Number')

    # if a is a csdl variable and is a scalar, expand it to match var shape
    if isinstance(a, Variable):
        if a.shape == (1,):
            from csdl import expand as csdl_expand
            a2 = csdl_expand(a, shape = var.shape)
        else:
            a2 = a
    else:
        a2 = a

    op = ops.exp_a(
        var,
        a=a2,
    )
    op.outs = (Output(
        None,
        op=op,
        shape=op.dependencies[0].shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)
    return op.outs[0]