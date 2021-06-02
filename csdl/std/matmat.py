from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
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
    if not (len(mat1.shape) == 2 and len(mat2.shape) == 2):
        raise ValueError(
            "Arguments must both be matrices (rank 2 tensors); {} has shape {}, and {} has shape {}"
            .format(
                mat1.name,
                mat1.shape,
                mat2.name,
                mat2.shape,
            ))
    op = ops.matmat(mat1, mat2)
    op.outs = [
        Output(
            None,
            op=op,
            shape=(mat1.shape[0], mat2.shape[1]),
        ),
    ]
    for out in op.outs:
        out.add_dependency_node(op)

    return op.outs[0]
