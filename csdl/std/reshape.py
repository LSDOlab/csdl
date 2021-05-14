from csdl.comps.reshape_comp import ReshapeComp
from csdl.core.variable import Variable


def reshape(expr: Variable, new_shape: tuple):
    '''
    This function reshapes the input into a new shape.

    Parameters
    ----------
    expr: Variable
        The Variable which you want to reshape

    new_shape: tuple[int]
        A tuple of ints specifying the new shape desired
    '''
    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")
    out = Variable()
    out.shape = new_shape
    out.add_dependency_node(expr)
    out.build = lambda: ReshapeComp(
        shape=expr.shape,
        in_name=expr.name,
        out_name=out.name,
        new_shape=out.shape,
        val=expr.val,
    )
    return out
