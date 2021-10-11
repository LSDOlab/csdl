from csdl.core.custom_operation import CustomOperation
from csdl.core.output import Output


def custom(*args, op: CustomOperation):
    for arg in args:
        if not isinstance(arg, Output):
            raise ValueError("Variable {} is not an Output".format(
                arg.name))
        if arg.name not in op.input_meta.keys():
            raise ValueError("Variable not found in CustomOperation")
        if arg.shape != op.input_meta[arg.name]:
            raise ValueError(
                "Variable shapes do not match for {}. Argument shape is {}, but CustomOperation has shape {}"
                .format(arg.name, arg.shape,
                        op.input_meta[arg.name].shape))
        # need to update metadata for arg based on op.input_meta or vice versa?
        op.add_dependency_node(arg)
    outs = []
    for _, meta in op.output_meta.items():
        outs.append(Output(
            **meta,
            op=op,
        ))

    if len(outs) == 1:
        return outs[0]
    else:
        return tuple(outs)
