import csdl.operations as ops
from csdl.core.variable import Variable
from csdl.core.output import Output
import numpy as np


def exp_a(a, var):
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    
    # if a is not a scalar,
    # a shape must match var shape
    if isinstance(a, np.ndarray):
        a_shape = a.shape
        if (a_shape != (1,)):
            if a_shape != var.shape:
                raise ValueError(f'Constant \'a\' must be a scalar or have same shape as \'{var}\'. {a_shape} != {var.shape}.')


    op = ops.exp_a(var, a=a)
    op.outs = (Output(
        None,
        op=op,
        shape=op.dependencies[0].shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)
    return op.outs[0]
