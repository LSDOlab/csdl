from csdl.comps.tensor_dot_product_comp import TensorDotProductComp
from csdl.comps.vector_inner_product_comp import VectorInnerProductComp

from csdl.core.variable import Variable
from csdl.core.operation import Operation
import numpy as np


def dot(expr1: Variable, expr2: Variable, axis=None):
    '''
    This can the dot product between two inputs.

    Parameters
    ----------
    expr1: Variable
        The first input for the dot product.

    expr2: Variable
        The second input for the dot product.

    axis: int
        The axis along which the dot product is taken. The axis must
        have an axis of 3.
    '''

    if not (isinstance(expr1, Variable) and isinstance(expr2, Variable)):
        raise TypeError("Arguments must both be Variable objects")
    out = Operation()
    out.add_dependency_node(expr1)
    out.add_dependency_node(expr2)

    if expr1.shape != expr2.shape:
        raise Exception("The shapes of the inputs must match!")

    print(len(expr1.shape))
    print(len(expr2.shape))

    if len(expr1.shape) == 1:
        out.build = lambda: VectorInnerProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shape=expr1.shape[0],
            in_vals=[expr1.val, expr2.val],
        )
    else:
        if expr1.shape[axis] != 3:
            raise Exception(
                "The specified axis must correspond to the value of 3 in shape"
            )
        else:
            out.shape = tuple(np.delete(list(expr1.shape), axis))

            out.build = lambda: TensorDotProductComp(
                in_names=[expr1.name, expr2.name],
                out_name=out.name,
                in_shape=expr1.shape,
                axis=axis,
                out_shape=out.shape,
                in_vals=[expr1.val, expr2.val],
            )
    return Variable(out)
