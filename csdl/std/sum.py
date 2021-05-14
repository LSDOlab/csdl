from csdl.comps.single_tensor_sum_comp import SingleTensorSumComp
from csdl.comps.multiple_tensor_sum_comp import MultipleTensorSumComp
from csdl.core.variable import Variable
from typing import List
import numpy as np


def sum(*summands: List[Variable], axes=None):
    '''
    This function can compute an elementwise or axiswise sum of
    a single or multiple inputs.

    Parameters
    ----------
    summands: Variable(s)
        The Variable(s) over which to take the sum

    axes: tuple[int]
        The axes along which the sum will be taken over

    '''

    out = Variable()
    for expr in summands:
        if not isinstance(expr, Variable):
            raise TypeError(expr, " is not an Variable object")
        out.add_dependency_node(expr)

    if axes == None:
        if len(summands) == 1:
            out.build = lambda: SingleTensorSumComp(
                in_name=summands[0].name,
                shape=summands[0].shape,
                out_name=out.name,
                val=summands[0].val,
            )
        else:
            out.shape = expr.shape
            out.build = lambda: MultipleTensorSumComp(
                in_names=[expr.name for expr in summands],
                shape=expr.shape,
                out_name=out.name,
                vals=[expr.val for expr in summands],
            )
    else:
        output_shape = np.delete(expr.shape, axes)
        out.shape = tuple(output_shape)

        if len(summands) == 1:
            out.build = lambda: SingleTensorSumComp(
                in_name=expr.name,
                shape=expr.shape,
                out_name=out.name,
                out_shape=out.shape,
                axes=axes,
                val=summands[0].val,
            )
        else:
            out.build = lambda: MultipleTensorSumComp(
                in_names=[expr.name for expr in summands],
                shape=expr.shape,
                out_name=out.name,
                out_shape=out.shape,
                axes=axes,
                vals=[expr.val for expr in summands],
            )
    return out
