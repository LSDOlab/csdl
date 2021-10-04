import csdl.operations as ops
from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.utils.decompose_shape_tuple import decompose_shape_tuple


def expand(var: Variable, shape: tuple, indices=None):
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")

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

    try:
        op = ops.expand(var, expand_indices=expand_indices)
    except:
        op = ops.expand(var)
    op.outs = (Output(
        None,
        op=op,
        shape=shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)

    if not var.shape == (1, ):
        if indices is None:
            raise ValueError(
                'If expanding something other than a scalar ' +
                'indices must be given')
        (
            _,
            _,
            _,
            in_shape,
            _,
        ) = decompose_shape_tuple(shape, expand_indices)

        if in_shape != var.shape:
            raise ValueError('Shape or indices is invalid')

    return op.outs[0]
