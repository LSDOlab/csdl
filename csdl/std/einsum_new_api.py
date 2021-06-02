from csdl.utils.einsum_utils import compute_einsum_shape, new_einsum_subscripts_to_string_and_list
import csdl.operations as ops
from csdl.core.variable import Variable
from csdl.core.output import Output
from typing import List


def einsum_new_api(*operands: List[Variable],
                   operation: List[tuple],
                   partial_format='dense'):
    '''
    The Einstein Summation function performs the equivalent of numpy.einsum using a new api

    Parameters
    ----------
    operands: Variables(s)
        The Variable(s) which you would like to perform an einsum with.

    subscripts: list[tuple]
        Specifies the subscripts for summation as a list of tuples

    partial_format: str
        Denotes whether to compute 'dense' partials or 'sparse' partials

    '''
    for expr in operands:
        if not isinstance(expr, Variable):
            raise TypeError(expr, " is not an Variable object")
    if partial_format is not 'dense' and partial_format is not 'sparse':
        raise ValueError("partial_format must be \'dense\' or \'sparse\'")

    scalar_output = False
    if len(operands) == len(operation):
        scalar_output = True
    operation_aslist, operation_string = new_einsum_subscripts_to_string_and_list(
        operation, scalar_output=scalar_output)

    op = ops.einsum(*operands,
                    subscripts=operation_string,
                    partial_format=partial_format)

    shape = compute_einsum_shape(operation_aslist,
                                 [expr.shape for expr in operands])
    op.outs = (Output(None, op=op, shape=shape), )
    for out in op.outs:
        out.add_dependency_node(op)

    return out
