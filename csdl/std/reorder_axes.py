from csdl.core.variable import Variable
from csdl.core.output import Output
import csdl.operations as ops
from typing import List
import numpy as np
from csdl.utils.reorder_axes_utils import compute_new_axes_locations


def reorder_axes(var: Variable, operation: str):
    '''
    The function reorders the axes of the input.

    **Parameters**

    var: Variable
        The Variable that will have its axes reordered.

    operation: str
        Specifies the subscripts for reordering as comma separated list of subscript labels.
        Ex: 'ijk->kij'

    '''
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")

    new_axes_locations = compute_new_axes_locations(
        var.shape, operation)
    op = ops.reorder_axes(
        var,
        operation=operation,
        new_axes_locations=new_axes_locations,
    )
    shape = tuple(var.shape[i] for i in new_axes_locations)

    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    return op.outs[0]
