from csdl.comps.single_tensor_average_comp import SingleTensorAverageComp
from csdl.comps.multiple_tensor_average_comp import MultipleTensorAverageComp
from csdl.core.variable import Variable
from typing import List
import numpy as np


def average(*operands: List[Variable], axes=None):
    '''
    This function can compute the average of a single input, multiple inputs, or
    along an axis.

    Parameters
    ----------

    operands: Variables
        The Variable(s) over which to take the average


    axes: tuple[int]
        Axes along which to take the average, default value is None

    '''

    out = Variable()
    for expr in operands:
        if not isinstance(expr, Variable):
            raise TypeError(expr, " is not an Variable object")
        out.add_dependency_node(expr)

    if axes == None:
        if len(operands) == 1:
            out.build = lambda: SingleTensorAverageComp(
                in_name=operands[0].name,
                shape=operands[0].shape,
                out_name=out.name,
                val=operands[0].val,
            )
        else:
            out.shape = expr.shape
            out.build = lambda: MultipleTensorAverageComp(
                in_names=[expr.name for expr in operands],
                shape=expr.shape,
                out_name=out.name,
                vals=[expr.val for expr in operands],
            )
    else:
        output_shape = np.delete(expr.shape, axes)
        out.shape = tuple(output_shape)

        if len(operands) == 1:
            out.build = lambda: SingleTensorAverageComp(
                in_name=operands[0].name,
                shape=operands[0].shape,
                out_name=out.name,
                out_shape=out.shape,
                axes=axes,
                val=operands[0].val,
            )
        else:
            out.build = lambda: MultipleTensorAverageComp(
                in_names=[expr.name for expr in operands],
                shape=expr.shape,
                out_name=out.name,
                out_shape=out.shape,
                axes=axes,
                vals=[expr.val for expr in operands],
            )
    return out
