try:
    from csdl.lang.model import Model
except ImportError:
    pass
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.subgraph import Subgraph
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.utils.find_promoted_name import find_promoted_name
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


def collect_user_declared_connections(
    model: 'Model',
    namespace: str = '',
) -> list[Tuple[str, str, str]]:
    """
    Collect connections declared by user and store connections using
    unique names only.
    Unique names are promoted variable names relative to main model.
    """
    user_declared_connections_by_namespace: list[Tuple[str, str,
                                                       str]] = []
    for s in model.subgraphs:
        m = s.submodel
        c = collect_user_declared_connections(
            m,
            prepend_namespace(namespace, s.name),
        )
        user_declared_connections_by_namespace.extend(c)
    for src, tgt in model.user_declared_connections:
        user_declared_connections_by_namespace.append(
            (src, tgt, namespace))
    return user_declared_connections_by_namespace


def map_promoted_to_declared_connections(
    user_declared_connections_by_namespace: list[Tuple[str, str, str]],
    promoted_to_unpromoted: Dict[str, Set[str]],
    unpromoted_to_promoted: Dict[str, str],
) -> Dict[Tuple[str, str], list[Tuple[str, str, str]]]:
    # prepend namespace to connections declared by user
    connections_full_path: list[Tuple[str, str]] = [
        (prepend_namespace(namespace,
                           src), prepend_namespace(namespace, tgt))
        for (src, tgt,
             namespace) in user_declared_connections_by_namespace
    ]
    # map connections using promoted (unique) names to names from
    # connections as declared by user and namespace in which connections
    # were declared
    promoted_to_declared_connections: Dict[Tuple[str, str],
                                           list[Tuple[str, str,
                                                      str]]] = dict()
    for (a, b) in zip(connections_full_path,
                      user_declared_connections_by_namespace):
        # get connection using promoted names
        key = (
            find_promoted_name(a[0], promoted_to_unpromoted,
                               unpromoted_to_promoted),
            find_promoted_name(a[1], promoted_to_unpromoted,
                               unpromoted_to_promoted),
        )
        if key in promoted_to_declared_connections.keys():
            promoted_to_declared_connections[key].append(b)
        else:
            promoted_to_declared_connections[key] = [b]
    return promoted_to_declared_connections
