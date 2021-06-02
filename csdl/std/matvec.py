from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def matvec(mat, vec):
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

    if not (isinstance(mat, Variable) and isinstance(vec, Variable)):
        raise TypeError("Arguments must both be Variable objects")
    if not (mat.shape[1] == vec.shape[0] and len(vec.shape) == 1):
        raise ValueError(
            "Arguments must both be a matrix with shape (n, m) and a vector with shape (m, ); {} has shape {}, and {} has shape {}"
            .format(
                mat.name,
                mat.shape,
                vec.name,
                vec.shape,
            ))
    op = ops.matvec(mat, vec)
    op.outs = [
        Output(
            None,
            op=op,
            shape=(mat.shape[0], ),
        ),
    ]
    for out in op.outs:
        out.add_dependency_node(op)

    return op.outs[0]
