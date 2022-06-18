try:
    from csdl.lang.model import Model
except ImportError:
    pass
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.subgraph import Subgraph
from csdl.utils.prepend_namespace import prepend_namespace
from typing import Set, Dict, Tuple



def find_longest_varpath_in_set(s: Set[str]) -> list[str]:
    varpaths = [st.rsplit('.') for st in s]
    return max(varpaths, key=len)


def find_unique_name(
    name: str,
    main_mpu: Dict[str, Set[str]],
    main_mup: Dict[str, str],
    namespace: str,
) -> str:
    """
    Find unique name of a variable in a connection specified by user.
    """
    full_name = prepend_namespace(namespace, name)
    if full_name in main_mpu.keys():
        return full_name
    if full_name in main_mup.keys():
        return main_mup[full_name]
    raise KeyError(
            "{} is not a user defined variable name in the model".
            format(name))



def check_source_name(model: 'Model', name):
    # name is promoted name
    if name in model.promoted_names_to_unpromoted_names.keys():
        # name is promoted name of source, not target
        if name in model.promoted_source_shapes.keys():
            return
    # name is unpromoted name
    if name in model.unpromoted_to_promoted.keys():
        # name is promoted name of source, not target
        if model.unpromoted_to_promoted[
                name] in model.promoted_source_shapes.keys():
            return
    raise KeyError(
        "{} is not a valid source variable name for connection. Source variable must be an Input or an Output."
    )


def check_target_name(model: 'Model', name):
    # name is promoted name
    if name in model.promoted_names_to_unpromoted_names.keys():
        # name is promoted name of target, not target
        if name in model.promoted_target_shapes.keys():
            return
    # name is unpromoted name
    if name in model.unpromoted_to_promoted.keys():
        # name is promoted name of target, not source
        if model.unpromoted_to_promoted[
                name] in model.promoted_target_shapes.keys():
            return
    raise KeyError(
        "{} is not a valid target variable name for connection. Target variable must be a DeclaredVariable."
    )



def collect_connections(
    model: 'Model',
    main_promoted_to_unpromoted: Dict[str, Set[str]],
    main_unpromoted_to_promoted: Dict[str, str],
    namespace: str = '',
) -> list[Tuple[str, str]]:
    """
    Collect connections declared by user and store connections using
    unique names only.
    Unique names are promoted variable names relative to main model.
    """
    connections: list[Tuple[str, str]] = []
    for s in model.subgraphs:
        c = collect_connections(
            s.submodel,
            main_promoted_to_unpromoted,
            main_unpromoted_to_promoted,
            prepend_namespace(namespace, s.name),
        )
        connections.extend(c)
    for a, b in model.user_declared_connections:
        check_source_name(model, a)
        check_target_name(model, b)
        src = find_unique_name(
            a,
            main_promoted_to_unpromoted,
            main_unpromoted_to_promoted,
            namespace,
        )
        tgt = find_unique_name(
            b,
            main_promoted_to_unpromoted,
            main_unpromoted_to_promoted,
            namespace,
        )
        connections.append((src, tgt))
    # return list of connections without duplicates
    # TODO: raise error on redundant connections, show all possible
    # unpromoted names for each connection
    return list(set(connections))