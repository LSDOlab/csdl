from csdl.utils.prepend_namespace import prepend_namespace
from typing import Set, Dict, Tuple, List


def collect_connections(
    model: 'Model',
    namespace: str = '',
) -> List[Tuple[str, str, str]]:
    '''
    Go through each model recursively and return connections.

    Parameters:
        model: current model to check for connections
        namespace: current namespace for model


    Returns:
        list with tuples (<source name>, <target name>, <namespace>)
        where 
        <source name> is the promoted name relative to system model
        <target name> is the promoted name relative to system model
        <namespace> is the model's namespace
    '''
    collected_connections = []

    # Recursive bottom model -> top model
    for s in model.subgraphs:
        submodel = s.submodel

        c = collect_connections(
            submodel,
            prepend_namespace(namespace, s.name),
        )
        collected_connections.extend(c)

    # Collect connections for current model 'model'
    for a, b in model.user_declared_connections:
        collected_connections.append((a, b, namespace))

    # return
    return collected_connections
