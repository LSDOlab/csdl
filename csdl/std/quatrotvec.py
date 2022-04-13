from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np


def quatrotvec(quat, vec):
    '''
    This function can compute a quaternion-vector multiplication, that results in a 
    rotated vector.

    **Parameters**

    quat: Variable
        The first input for the quat-vec multiplication

    vec: Variable
        The second input for the quat-vec multiplication

    '''

    if not (isinstance(quat, Variable) and isinstance(vec, Variable)):
        raise TypeError("Arguments must both be Variable objects")
    if not (quat.shape[:-1] == vec.shape[:-1]):
        raise ValueError(
            "Arguments must both be matrices (rank 2 tensors); {} has shape {}, and {} has shape {}"
            .format(
                quat.name,
                quat.shape,
                vec.name,
                vec.shape,
            ))
            
    op = ops.quatrotvec(quat, vec)
    op.outs = [
        Output(
            None,
            op=op,
            shape=vec.shape,
        ),
    ]
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
