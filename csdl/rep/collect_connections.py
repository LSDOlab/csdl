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
        "{} is not a user defined variable name in the model".format(
            name))


def check_source_name(
    model: 'Model',
    name: str,
    promoted_source_names: Set[str],
):
    # name is promoted name
    if name in model.promoted_to_unpromoted.keys():
        # name is promoted name of source, not target
        if name in promoted_source_names:
            return
    # name is unpromoted name
    if name in model.unpromoted_to_promoted.keys():
        # name is promoted name of source, not target
        if model.unpromoted_to_promoted[name] in promoted_source_names:
            return
    raise KeyError(
        "\'{}\' is not a valid source variable name for connection. Source variable must be an Input or an Output of the model or promoted from its submodels."
        .format(name))


def collect_promoted_source_names_lower_levels(
    model: 'Model',
    promotes: list[str] | None = None,
    namespace: str = '',
) -> Set[str]:
    promoted_names = set()
    for s in model.subgraphs:
        m = s.submodel
        collect_promoted_source_names_lower_levels(
            m,
            promotes=s.promotes,
            namespace=prepend_namespace(namespace, s.name),
        )
    promoted_names.update([
        name if promotes is None or name in promotes else
        prepend_namespace(namespace, name)
        for name in model.promoted_source_shapes.keys()
    ])
    return promoted_names


def collect_promoted_target_names_lower_levels(
    model: 'Model',
    promotes: list[str] | None = None,
    namespace: str = '',
) -> Set[str]:
    promoted_names = set()
    for s in model.subgraphs:
        m = s.submodel
        collect_promoted_target_names_lower_levels(
            m,
            promotes=s.promotes,
            namespace=prepend_namespace(namespace, s.name),
        )
    promoted_names.update([
        name if promotes is None or name in promotes else
        prepend_namespace(namespace, name)
        for name in model.promoted_target_shapes.keys()
    ])
    return promoted_names


def check_target_name(
    model: 'Model',
    name: str,
    promoted_target_names: Set[str],
):
    # name is promoted name
    if name in model.promoted_to_unpromoted.keys():
        # name is promoted name of target, not target
        if name in promoted_target_names:
            return
    # name is unpromoted name
    if name in model.unpromoted_to_promoted.keys():
        # name is promoted name of target, not source
        if model.unpromoted_to_promoted[name] in promoted_target_names:
            return
    raise KeyError(
        "\'{}\' is not a valid target variable name for connection. Target variable must be a DeclaredVariable of the model or promoted from its submodels."
        .format(name))


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
    promoted_source_names: Set[str] = set(
        model.promoted_source_shapes.keys())
    promoted_target_names: Set[str] = set(
        model.promoted_target_shapes.keys())
    for s in model.subgraphs:
        m = s.submodel
        c = collect_connections(
            m,
            main_promoted_to_unpromoted,
            main_unpromoted_to_promoted,
            prepend_namespace(namespace, s.name),
        )
        connections.extend(c)
        promoted_source_names.update([
            name if s.promotes is not None and name in s.promotes else
            prepend_namespace(s.name, name)
            for name in m.promoted_source_shapes.keys()
        ])
        promoted_target_names.update([
            name if s.promotes is not None and name in s.promotes else
            prepend_namespace(s.name, name)
            for name in m.promoted_target_shapes.keys()
        ])
    connections.extend(model.user_declared_connections)
    for a, b in model.user_declared_connections:
        check_source_name(model, a, promoted_source_names)
        check_target_name(model, b, promoted_target_names)
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
