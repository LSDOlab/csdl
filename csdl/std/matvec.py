from csdl.comps.matvec_comp import MatVecComp
from csdl.core.variable import Variable
from typing import List
import numpy as np


def matvec(mat1, vec1):
    '''
    This function can compute a matrix-vector multiplication, similar to the
    numpy counterpart.

    Parameters
    ----------
    mat1: Variable
        The matrix needed for the matrix-vector multiplication

    vec1: Variable
        The vector needed for the matrix-vector multiplication

    '''
    if not (isinstance(mat1, Variable) and isinstance(vec1, Variable)):
        raise TypeError("Arguments must both be Variable objects")
    out = Variable()
    out.add_dependency_node(mat1)
    out.add_dependency_node(vec1)

    if mat1.shape[1] == vec1.shape[0] and len(vec1.shape) == 1:

        out.shape = (mat1.shape[0], )

        out.build = lambda: MatVecComp(
            in_names=[mat1.name, vec1.name],
            out_name=out.name,
            in_shapes=[mat1.shape, vec1.shape],
            in_vals=[mat1.val, vec1.val],
        )

    else:
        raise Exception("Cannot multiply: ", mat1.shape, "by", vec1.shape)
    return out
