from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np
import scipy.sparse


def sparsematmat(mat, sparse_mat=None):
    '''
    This function can compute a quaternion-vector multiplication, that results in a 
    rotated vector.

    **Parameters**

    quat: Variable
        The first input for the quat-vec multiplication

    vec: Variable
        The second input for the quat-vec multiplication

    '''

    if not isinstance(mat, Variable):
        raise TypeError("Argument must be a Variable object")

    if not scipy.sparse.issparse(sparse_mat):
        raise TypeError('Sparse_mat must a sparse matrix!')

    if not (mat.shape[0] == sparse_mat.shape[1]):
        raise ValueError(
            "Arguments must both be matrices (rank 2 tensors); {} has shape {}, and {} has shape {}"
            .format(
                'sparse_mat',
                sparse_mat.shape,
                mat.name,
                mat.shape,
            ))
            
    op = ops.sparsematmat(mat, sparse_mat=sparse_mat)
    op.outs = [
        Output(
            None,
            op=op,
            shape=(sparse_mat.shape[0], mat.shape[1]),
        ),
    ]
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
