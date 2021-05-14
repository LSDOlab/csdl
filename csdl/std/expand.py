from csdl.comps.array_expansion_comp import ArrayExpansionComp
from csdl.comps.scalar_expansion_comp import ScalarExpansionComp
from csdl.core.variable import Variable
from csdl.utils.decompose_shape_tuple import decompose_shape_tuple


def expand(expr: Variable, shape: tuple, indices=None):

    if not isinstance(expr, Variable):
        raise TypeError(expr, " is not an Variable object")

    if indices is not None:
        if not isinstance(indices, str):
            raise TypeError(indices, " is not a str or None")

        if '->' not in indices:
            raise ValueError(indices, " is invalid")

    if indices is not None:
        in_indices, out_indices = indices.split('->')
        expand_indices = []
        for i in range(len(out_indices)):
            index = out_indices[i]

            if index not in in_indices:
                expand_indices.append(i)

    out = Variable()
    out.shape = shape
    out.add_dependency_node(expr)

    if not expr.shape == (1, ):
        if indices is None:
            raise ValueError('If expanding something other than a scalar ' +
                             'indices must be given')
        (
            _,
            _,
            _,
            in_shape,
            _,
            _,
        ) = decompose_shape_tuple(shape, expand_indices)

        if in_shape != expr.shape:
            raise ValueError('Shape or indices is invalid')

        out.build = lambda: ArrayExpansionComp(
            shape=shape,
            expand_indices=expand_indices,
            in_name=expr.name,
            out_name=out.name,
            val=expr.val,
        )
    else:
        out.build = lambda: ScalarExpansionComp(
            shape=shape,
            in_name=expr.name,
            out_name=out.name,
        )
    return out
