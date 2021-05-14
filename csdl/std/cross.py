from csdl.comps.cross_product_comp import CrossProductComp
from csdl.core.variable import Variable


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
    out = Variable()
    out.add_dependency_node(in1)
    out.add_dependency_node(in2)

    if in1.shape != in2.shape:
        raise Exception("The shapes of the inputs must match!")
    else:
        out.shape = in1.shape

    if in1.shape[axis] != 3:
        raise Exception(
            "The specified axis must correspond to the value of 3 in shape")

    out.build = lambda: CrossProductComp(
        shape=in1.shape,
        in1_name=in1.name,
        in2_name=in2.name,
        out_name=out.name,
        axis=axis,
        in1_val=in1.val,
        in2_val=in2.val,
    )
    return out
