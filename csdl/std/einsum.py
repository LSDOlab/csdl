import csdl.operations as ops
from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.utils.einsum_utils import compute_einsum_shape, einsum_subscripts_tolist
from typing import List


def einsum(*operands: List[Variable],
           subscripts: str,
           partial_format='dense'):
    '''
    The Einstein Summation function performs the equivalent of numpy.einsum

    **Parameters**

    operands: Variable(s)
        The Variable(s) which you would like to perform an einsum with.

    subscripts: str
        Specifies the subscripts for summation as comma separated list of subscript labels

    partial_format: str
        Denotes whether to compute 'dense' partials or 'sparse' partials

    '''
    for expr in operands:
        if not isinstance(expr, Variable):
            raise TypeError(expr, " is not an Variable object")
    if partial_format is not 'dense' and partial_format is not 'sparse':
        raise ValueError(
            "partial_format must be \'dense\' or \'sparse\'")

    op = ops.einsum(*operands,
                    subscripts=subscripts,
                    partial_format=partial_format)

    operation_aslist = einsum_subscripts_tolist(subscripts)
    shape = compute_einsum_shape(operation_aslist,
                                 [expr.shape for expr in operands])
    op.outs = (Output(None, op=op, shape=shape), )
    for out in op.outs:
        out.add_dependency_node(op)

    return op.outs[0]
