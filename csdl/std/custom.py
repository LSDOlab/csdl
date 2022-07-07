from csdl.lang.custom_operation import CustomOperation
from csdl.lang.variable import Variable
from csdl.lang.output import Output
from collections import OrderedDict
from csdl.utils.gen_hex_name import gen_hex_name


def custom(*args, op: CustomOperation):
    op.define()
    if len(args) < len(op.input_meta):
        raise TypeError(
            "Too few arguments for CustomOperation; expected {}, got {}"
            .format(len(op.input_meta), len(args)))
    if len(args) > len(op.input_meta):
        raise TypeError(
            "Too many arguments for CustomOperation; expected {}, got {}"
            .format(len(op.input_meta), len(args)))

    for arg, key in zip(args, op.input_meta.keys()):
        if not isinstance(arg, Variable):
            raise ValueError(
                "Variable {} is not a CSDL Variable".format(arg.name))
        # KLUDGE: outer name must match inner name
        if arg.name != key:
            raise ValueError(
                "Variable {} does not match {}. Perhaps the arguments are out of order?"
                .format(arg.name, key))
        if arg.shape != op.input_meta[arg.name]['shape']:
            raise ValueError(
                "Variable shapes do not match for {}. Argument shape is {}, but CustomOperation has shape {}"
                .format(arg.name, arg.shape,
                        op.input_meta[arg.name]['shape']))
        # need to update metadata for arg based on op.input_meta or vice versa?
        op.add_dependency_node(arg)
        op.input_meta[arg.name]['val'] *= 0
        op.input_meta[arg.name]['val'] += arg.val
    outs = []
    # KLUDGE: keep names simple for now
    for name, meta in op.output_meta.items():
        out = Output(
            name,
            **meta,
            op=op,
        )
        outs.append(out)

    if len(outs) == 1:
        return outs[0]
    else:
        return tuple(outs)
