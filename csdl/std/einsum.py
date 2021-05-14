from csdl.comps.einsum_comp_dense_derivs import EinsumComp
from csdl.comps.einsum_comp_sparse_derivs import SparsePartialEinsumComp
from csdl.utils.einsum_utils import compute_einsum_shape, einsum_subscripts_tolist
from csdl.core.variable import Variable
from typing import List


def einsum(*operands: List[Variable], subscripts: str, partial_format='dense'):
    '''
    The Einstein Summation function performs the equivalent of numpy.einsum

    Parameters
    ----------
    operands: Variable(s)
        The Variable(s) which you would like to perform an einsum with.

    subscripts: str
        Specifies the subscripts for summation as comma separated list of subscript labels

    partial_format: str
        Denotes whether to compute 'dense' partials or 'sparse' partials

    '''
    out = Variable()
    for expr in operands:
        if not isinstance(expr, Variable):
            raise TypeError(expr, " is not an Variable object")
        out.add_dependency_node(expr)
    operation_aslist = einsum_subscripts_tolist(subscripts)
    shape = compute_einsum_shape(operation_aslist,
                                 [expr.shape for expr in operands])
    out.shape = shape

    if partial_format == 'dense':
        out.build = lambda: EinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=out.name,
            operation=subscripts,
            out_shape=shape,
            in_vals=[expr.val for expr in operands],
        )
    elif partial_format == 'sparse':
        out.build = lambda: SparsePartialEinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=out.name,
            operation=subscripts,
            out_shape=shape,
            in_vals=[expr.val for expr in operands],
        )
    else:
        raise Exception('partial_format should be either dense or sparse')
    return out
