from csdl.comps.matmat_comp import MatMatComp
from csdl.core.variable import Variable
from typing import List
import numpy as np


def matmat(mat1, mat2):
    '''
    This function can compute a matrix-matrix multiplication, similar to the
    numpy counterpart.

    Parameters
    ----------
    mat1: Variable
        The first input for the matrix-matrix multiplication

    mat2: Variable
        The second input for the matrix-matrix multiplication

    '''

    if not (isinstance(mat1, Variable) and isinstance(mat2, Variable)):
        raise TypeError("Arguments must both be Variable objects")
    out = Variable()
    out.add_dependency_node(mat1)
    out.add_dependency_node(mat2)

    if mat1.shape[1] == mat2.shape[0] and len(mat2.shape) == 2:
        # Compute the output shape if both inputs are matrices
        out.shape = (mat1.shape[0], mat2.shape[1])

        out.build = lambda: MatMatComp(
            in_names=[mat1.name, mat2.name],
            out_name=out.name,
            in_shapes=[mat1.shape, mat2.shape],
            in_vals=[mat1.val, mat2.val],
        )

    elif mat1.shape[1] == mat2.shape[0] and len(mat2.shape) == 1:
        out.shape = (mat1.shape[0], 1)

        mat2_shape = (mat2.shape[0], 1)

        out.build = lambda: MatMatComp(
            in_names=[mat1.name, mat2.name],
            out_name=out.name,
            in_shapes=[mat1.shape, mat2_shape],
            in_vals=[mat1.val, mat2.val.reshape(mat2_shape)],
        )
    else:
        raise Exception("Cannot multiply: ", mat1.shape, "by", mat2.shape)
    return out
