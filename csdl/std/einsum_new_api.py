from csdl.comps.einsum_comp_dense_derivs import EinsumComp
from csdl.comps.einsum_comp_sparse_derivs import SparsePartialEinsumComp
from csdl.utils.einsum_utils import compute_einsum_shape, new_einsum_subscripts_to_string_and_list
from csdl.core.variable import Variable
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
    out = Variable()
    for expr in operands:
        if not isinstance(expr, Variable):
            raise TypeError(expr, " is not an Variable object")
        out.add_dependency_node(expr)
    scalar_output = False
    if len(operands) == len(operation):
        scalar_output = True
    operation_aslist, operation_string = new_einsum_subscripts_to_string_and_list(
        operation, scalar_output=scalar_output)

    shape = compute_einsum_shape(operation_aslist,
                                 [expr.shape for expr in operands])
    out.shape = shape

    if partial_format == 'dense':
        out.build = lambda: EinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=out.name,
            operation=operation_string,
            out_shape=shape,
            in_vals=[expr.val for expr in operands],
        )
    elif partial_format == 'sparse':
        out.build = lambda: SparsePartialEinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=out.name,
            operation=operation_string,
            out_shape=shape,
            in_vals=[expr.val for expr in operands],
        )
    else:
        raise Exception('partial_format should be either dense or sparse')
    return out
