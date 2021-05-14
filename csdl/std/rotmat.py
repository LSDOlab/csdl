from csdl.comps.rotation_matrix_comp import RotationMatrixComp
from csdl.core.variable import Variable


def rotmat(expr: Variable, axis: str):
    '''
    This function creates a rotation matrix depending on the input value and the axis.

    Parameters
    ----------
    expr: Variable
        The value which determines by how much the rotation matrix

    axis: str
        The axis along which the rotation matrix should rotate. Can we specified
        with: 'x' , 'y' , or 'z'.

    '''
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.add_dependency_node(expr)

    if expr.shape == (1, ):
        out.shape = (3, 3)

    else:
        out.shape = expr.shape + (3, 3)

    out.build = lambda: RotationMatrixComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        axis=axis,
        val=expr.val,
    )
    return out
