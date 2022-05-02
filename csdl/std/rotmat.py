from csdl.lang.variable import Variable
from csdl.lang.output import Output
import csdl.operations as ops


def rotmat(var: Variable, axis: str):
    '''
    This function creates a rotation matrix depending on the input value and the axis.

    **Parameters**

    expr: Variable
        The value which determines by how much the rotation matrix

    axis: str
        The axis along which the rotation matrix should rotate. Can we specified
        with: 'x' , 'y' , or 'z'.

    '''
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    op = ops.rotmat(var, axis=axis)

    if var.shape == (1, ):
        shape = (3, 3)
    else:
        shape = var.shape + (3, 3)

    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
