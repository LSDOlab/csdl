try:
    from csdl.lang.model import Model
except ImportError:
    pass
try:
    from csdl.lang.output import Output
except ImportError:
    pass
try:
    from csdl.lang.input import Input
except ImportError:
    pass
from typing import List, Tuple


def collect_connections(
    model: 'Model',
    prefix: str | None = None,
) -> List[Tuple[str, str]]:
    """
    Collect connections between models at all levels within the
    model heirarchy and store connections using unique, promoted
    variable names.
    """
    connections = []
    for s in model.subgraphs:
        m = s.submodel
        connections.extend(collect_connections(m, prefix=s.name))

    # prevent user from connecting an input/output to a declared
    # variable within the same model; no reason to ever do this
    for (a, b) in model.user_declared_connections:
        io: list[Input | Output] = []
        io.extend(model.inputs)
        io.extend(model.registered_outputs)
        if a in [x.name for x in io
                 ] and b in [x.name for x in model.declared_variables]:
            raise KeyError(
                "Cannot connect source {} to sink {} as both variables are defined within this model."
                .format(a, b))

    if prefix is None:
        connections.extend(model.connections)
    else:
        connections.extend([(prefix + '.' + a, prefix + '.' + b)
                            for (a, b) in model.connections])
    return connections
