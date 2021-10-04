import csdl.operations as ops
from csdl.core.variable import Variable
from csdl.core.output import Output


def arccos(var):
    if not isinstance(var, Variable):
        raise TypeError(var, " is not an Variable object")
    op = ops.arccos(var)
    op.outs = (Output(
        None,
        op=op,
        shape=op.dependencies[0].shape,
    ), )
    # for out in op.outs:
    #         out.add_dependency_node(op)
    return op.outs[0]
