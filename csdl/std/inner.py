from csdl.comps.tensor_inner_product_comp import TensorInnerProductComp
from csdl.comps.vector_inner_product_comp import VectorInnerProductComp

from csdl.core.variable import Variable
from typing import List
import numpy as np


def inner(expr1: Variable, expr2: Variable, axes=None):
    '''
    This can the dot product between two inputs.

    Parameters
    ----------
    expr1: Variable
        The first input for the inner product.

    expr2: Variable
        The second input for the inner product.

    axes: tuple[int]
        The axes along which the inner product is taken.
    '''
    if not isinstance(expr1, Variable):
        raise TypeError(expr1, " is not an Variable object")
    elif not isinstance(expr2, Variable):
        raise TypeError(expr2, " is not an Variable object")

    out = Variable()
    out.add_dependency_node(expr1)
    out.add_dependency_node(expr2)

    if len(expr1.shape) == 1 and len(expr2.shape) == 1:
        out.build = lambda: VectorInnerProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shape=expr1.shape[0],
            in_vals=[expr1.val, expr2.val],
        )

    else:
        new_in0_shape = np.delete(list(expr1.shape), axes[0])
        new_in1_shape = np.delete(list(expr2.shape), axes[1])
        out.shape = tuple(np.append(new_in0_shape, new_in1_shape))

        out.build = lambda: TensorInnerProductComp(
            in_names=[expr1.name, expr2.name],
            out_name=out.name,
            in_shapes=[expr1.shape, expr2.shape],
            axes=axes,
            out_shape=out.shape,
            in_vals=[expr1.val, expr2.val],
        )
    return out
