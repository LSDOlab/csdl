from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops


def cross(in1, in2, axis: int):
    '''
    This can the cross product between two inputs.

    Parameters
    ----------
    in1: Variable
        The first input for the cross product.

    in2: Variable
        The second input for the cross product.

    axis: int
        The axis along which the cross product is taken. The axis specified must
        have a value of 3.
    '''

    if not (isinstance(in1, Variable) and isinstance(in2, Variable)):
        raise TypeError("Arguments must both be Variable objects")
    if in1.shape != in2.shape:
        raise ValueError("The shapes of the inputs must match!")
    if in1.shape[axis] != 3:
        raise ValueError(
            "The specified axis must correspond to the value of 3 in shape")

    op = ops.cross(in1, in2, axis=axis)
    op.outs = (Output(
        None,
        op=op,
        shape=op.dependencies[0].shape,
    ), )
    for out in op.outs:
        out.add_dependency_node(op)

    return op.outs[0]
