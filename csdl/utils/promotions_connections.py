from csdl.core.model import Model
from csdl.core.output import Output
from csdl.utils.typehints import Shape
from typing import Callable, Dict, List, Tuple, Any, TypeVar, Set
from copy import copy
import networkx as nx

T = TypeVar('T', bound=Any)
U = TypeVar('U', bound=Any)
Model = TypeVar('Model', bound=Model)

# declared variables no longer depend on subgraphs
# after promotions are evaluated, each output promoted from a subgraph
# will replace declared variable with same name and shape in parent
# model, and new output node will depend on subgraph
# note: if promotions are successful, output from subgraph and declared
# variable in parent model with same name will have same shape; no need
# to check twice


def find_keys_with_matching_values(
    a: Dict[T, U],
    b: Dict[T, U],
    error: Callable[[T, T, U, U], BaseException] | None = None,
) -> Set[T]:
    """
    Find keys common to two dictionaries with matching values. Keys must
    be of a common type, and values must be of a common second type. If
    error provided, throw error if any values mismatch for any key
    common to both dictionaries.
    """
    c: Dict[T, U] = dict()
    d: Dict[T, U] = dict()
    matching_keys: Set[T] = set()

    # ensure that keys match when iterating over dictionary items
    for k in a.keys():
        if k in b.keys():
            c[k] = a[k]
            d[k] = b[k]

    # iterate over dictionary items with matching keys
    for (k1, v1), (k2, v2) in zip(c.items(), d.items()):
        if v1 == v2:
            matching_keys.add(k1)
        elif error is not None:
            raise error(k1, k2, v1, v2)
        else:
            continue
    return matching_keys


def find_names_with_matching_shapes(
    a: Dict[str, Shape],
    b: Dict[str, Shape],
) -> Set[str]:
    """
    Find names of variables whose shapes match. If any shapes mismatch,
    throw an error.
    """
    return find_keys_with_matching_values(
        a,
        b,
        error=lambda name1, name2, shape1, shape2: ValueError(
            "Shapes do not match; {} has shape {}, and {} has shape {}".
            format(name1, name2, shape1, shape2)))


def prepend_namespaces(
        subgraph: Subgraph,
        promoted_to_unpromoted: Dict[str,
                                     Set[str]]) -> Dict[str, Set[str]]:
    promoted_to_unpromoted2: Dict[str, Set[str]] = dict()
    for k, v in promoted_to_unpromoted.items():
        promoted_to_unpromoted2[subgraph.name + '.' + k] = {
            subgraph.name + '.' + x
            for x in v
        }
    return promoted_to_unpromoted2


def gather_variables_to_promote_no_futher(
        model: Model,
        locally_defined_promotable_names: Set[str]) -> Dict[str, Shape]:
    variables_to_promote_no_further = {
        x.name: x.shape
        for x in model.inputs + model.outputs + model.declared_variables
    }
    for k in locally_defined_promotable_names:
        variables_to_promote_no_further.pop(k)
    return variables_to_promote_no_further


def collect_promoted_variable_names(
    model: Model,
    main: bool = True,
    promotes: List[str] | None = None,
) -> Tuple[Dict[str, Shape], Dict[str, Shape]]:
    """
    Collect variables promoted from submodels to `model` and return
    promotions specified by `promotes`.
    Updates `model.promoted_sources` and `model.promoted_sinks` so that
    CSDL can process connections later. Also updates `model.connections`
    whenever a connection is formed due to a promoted source/sink pair
    having the same name and shape.
    """
    promoted_sources: Dict[str, Shape] = dict()
    promoted_sinks: Dict[str, Shape] = dict()

    for s in model.subgraphs:
        m = s.submodel
        p1, p2, pu = collect_promoted_variable_names(m, main=False)

        # required for checking if user has specified promotions that
        # result in cycles between models
        model.promoted_to_unpromoted = prepend_namespaces(s, pu)

        # Rule: each source (input or output) name promoted to this
        # level must be unique
        duplicate_source_names = promoted_sources.keys() & p1.keys()
        if duplicate_source_names != set():
            raise KeyError(
                "Cannot promote two inputs, two outputs, or an input and an output with the same name. Please check the variables with the following promoted paths: {}"
                .format(list(duplicate_source_names)))
        promoted_sources.update(p1)

        # Rule: all sinks (declared variable) with a common name
        # promoted to this level must have the same shape
        _ = find_names_with_matching_shapes(promoted_sinks, p2)
        promoted_sinks.update(p2)

    # Rule: if a source and a sink promoted to this level have matching
    # name and shape, then they form a connection
    connected_sinks = find_names_with_matching_shapes(
        promoted_sources, promoted_sinks)

    # Rule: If a declared variable is connected automatically by
    # promotion, then any further promotion will be applied to the
    # source (input or output) connected to the sink
    for c in connected_sinks.keys():
        promoted_sinks.pop(c)

    # Rule: each source (input or output) name specified in this level
    # must not match the name of any source promoted to this level
    locally_defined_sources: Dict[str, Shape] = {
        x.name: x.shape
        for x in model.inputs + model.registered_outputs
    }
    locally_defined_promotable_sources: Dict[str, Shape] = dict(
        filter(
            lambda x: x[0] in set(promotes)
            if promotes is not None else True,
            locally_defined_sources.items()))
    duplicate_source_names = promoted_sources.keys(
    ) & locally_defined_promotable_sources.keys()
    if duplicate_source_names != set():
        raise KeyError(
            "Cannot promote two inputs, two outputs, or an input and an output with the same name. Please check the following: {}"
            .format(list(duplicate_source_names)))

    # Rule: all sinks (declared variable) with a common name specified
    # in this level must have the same shape as any sink promoted to
    # this level with the same name
    locally_defined_sinks: Dict[str, Shape] = {
        x.name: x.shape
        for x in model.declared_variables
    }
    locally_defined_promotable_sinks: Dict[str, Shape] = dict(
        filter(
            lambda x: x[0] in set(promotes)
            if promotes is not None else True,
            locally_defined_sinks.items()))

    # ensure user has specified valid promotions only
    locally_defined_promotable_names: Set[
        str] = locally_defined_promotable_sources.keys(
        ) | locally_defined_promotable_sinks.keys()
    valid_promotes: Set[
        str] = locally_defined_promotable_names if promotes is None else set(
            promotes)
    invalid_promotes: Set[str] = set(promotes) - (
        valid_promotes) if promotes is not None else set()
    if invalid_promotes != set():
        # TODO: include model _object_ name in error message
        raise KeyError(
            # "The following variables are not valid for promotion in model with {} {} of type {}: {}".format(
            # "autogenerated name" if autogenerated_name is True
            # else "user-specified name", name,
            "The following variables are not valid for promotion in model of type {}: {}"
            .format(type(model).__name__),
            invalid_promotes)
    _ = find_names_with_matching_shapes(
        promoted_sinks, locally_defined_promotable_sinks)

    # update names to shape mappings for variables promoted to parent
    # model
    promoted_sources.update(locally_defined_promotable_sources)
    promoted_sources.update(locally_defined_promotable_sinks)

    # store a copy of promotions in model for parent models to access,
    # but not modify unless specifically accessing this model's
    # attributes to ensure promoted variable names are unique
    variables_to_promote_no_further = gather_variables_to_promote_no_futher(
        model, locally_defined_promotable_names)
    model.promoted_sources = copy(promoted_sources).update(
        variables_to_promote_no_further)
    model.promoted_sinks = copy(promoted_sources).update(
        variables_to_promote_no_further)

    # get the variables that will be promoted no further for parent
    # model to update its map from promoted paths to unpromoted paths
    for k in variables_to_promote_no_further.keys():
        model.promoted_to_unpromoted[k] = {k}

    if main is True:
        model.promoted_sources.update(sources)
        model.promoted_sinks.update(sinks)

    return promoted_sources, promoted_sinks


def collect_promoted_variable_paths(
    model: Model,
    prefix: str | None,
) -> List[str]:
    """
    Collect promoted paths for all variables to check for valid
    connections.
    This function should only ever be called once on the main model
    after promotions have been resolved.
    """
    a = []
    for s in model.subgraphs:
        a.extend([
            prefix + '.' + x if prefix is not None else x
            for x in collect_promoted_variable_paths(
                s.submodel,
                prefix=s.name,
            )
        ])
    return a + list(model.promoted_sources.keys()) + list(
        model.promoted_sinks.keys())


def share_namespace(a: Set[str], b: Set[str]):
    """
    Check if two sets of promoted names create a cycle
    """
    cycles_between_models: List[Tuple[str, str]] = []
    for x in a:
        for y in b:
            if x.rpartition('.')[0] == y.rpartition('.')[0]:
                cycles_between_models.append((x, y))
    num_cycles = len(cycles_between_models)
    if num_cycles > 0:
        raise KeyError(
            "Connections resulting from user-specified PROMOTIONS form the following {} cycles between variables: {}. "
            "CSDL does not support cyclic connections between models "
            "to describe coupling between models. "
            "To describe coupling between models, use an implicit operation. "
            "Up to {} implicit operations will be required to eliminate this "
            "error.".format(num_cycles, cycles_between_models,
                            num_cycles))


def check_for_promotion_induced_cycles(model: Model):
    for k1, s1 in model.promoted_to_unpromoted.items():
        for k2, s2 in model.promoted_to_unpromoted.items():
            if k1 != k2:
                share_namespace(s1, s2)


# if variables with different names are connected, then their relative
# promoted and unprmoted paths refer to the same variable
def promote_automatically_named_variables(
        model: Model) -> Dict[str, Shape]:
    """
    Promote automatically named variables to main model.
    Call once on main model at any time. Resolving other promotions will
    not affect the behavior or be affected by the behavior of this
    function.
    """
    graph = model.intermediate_representation.graph
    return {
        x.name: x.shape
        for x in list(
            filter(
                lambda x: isinstance(Output) and x not in set(
                    model.registered_outputs),
                graph.nodes(),
            ))
    }


def collect_connections(
        model: Model,
        prefix: str | None = None) -> List[Tuple[str, str]]:
    """
    Collect connections between models and store connections using
    unique, promoted variable names.
    """
    connections = []
    for s in model.subgraphs:
        m = s.submodel
        connections.extend(collect_connections(m, prefix=s.name))
    connections.extend([(prefix + '.' + a, prefix + '.' + b)
                        for (a, b) in m.connections])
    return connections


def detect_cycles_in_connections(connections: List[Tuple[str, str]], ):
    """
    Detect cycles formed from user-specified connections. If there are
    no cycles, return an acyclic graph with connections as nodes.
    """
    g = nx.DiGraph()
    g.add_edges_from(connections)
    cycles_between_models: List[str] = list(nx.simple_cycles(g))
    num_cyles = len(cycles_between_models)
    if num_cyles > 0:
        raise KeyError(
            "Connections resulting from user-specified CONNECTIONS form the following {} cycles between variables: {}. "
            "CSDL does not support cyclic connections between models "
            "to describe coupling between models. "
            "To describe coupling between models, use an implicit operation. "
            "Up to {} implicit operations will be required to eliminate this "
            "error.".format(num_cyles, cycles_between_models,
                            num_cyles))


def issue_user_specified_connections(model: Model):
    """
    Issue connections after all promotions have been resolved and
    connections formed due to promotions have been issued. User allowed
    to specify connections using relative promoted or relative unpromoted names. Stores
    connections in model using relative promoted names only.
    """
    for s in model.subgraphs:
        m = s.submodel
        issue_user_specified_connections(m)
        promoted_user_declared_connections: List[Tuple[str, str]] = []
        for (a, b) in m.user_declared_connections:
            if a in m.promoted_to_unpromoted.keys():
                promoted_a = a
            elif a in m.unpromoted_to_promoted.keys():
                promoted_a = m.unpromoted_to_promoted[a]
            else:
                raise KeyError(
                    "Variable {} is not a valid source (input or output) for conenction."
                    .format(a))
            if b in m.promoted_to_unpromoted.keys():
                promoted_b = b
            elif b in m.unpromoted_to_promoted.keys():
                promoted_b = m.unpromoted_to_promoted[b]
            else:
                raise KeyError(
                    "Variable {} is not a valid sink (declared variable) for conenction."
                    .format(a))

            promoted_user_declared_connections.append(
                (promoted_a, promoted_b))

        m.connections = list(
            set(m.connections + promoted_user_declared_connections))


# TODO:
def flatten_graph(model: Model):
    """
    Create a graph representing the model that contains only Variable
    and Operation nodes. No Subgraph nodes are present in the flattened
    graph. Model namespaces are preserved. All remaining
    `DeclaredVariable` nodes are replaced with `Input` nodes.
    """
    pass


# if __name__ == '__main__':

#     # This is what goes at the end of the middle end, prior to
#     # optimizing the intermediate representations

#     main_model = Model()

#     _, _ = collect_promoted_variable_names(main_model)
#     check_for_promotion_induced_cycles(main_model)

#     unpromoted_to_promoted = dict()
#     for k, v in main_model.promoted_to_unpromoted:
#         for n in v:
#             unpromoted_to_promoted[n] = k

#     issue_user_specified_connections(main_model)
#     connections = collect_connections(main_model)
#     detect_cycles_in_connections(connections)
#     # TODO: flatten graph
#     # now that intermediate representation is built and all namespaces
#     # are resolved, flatten graph and store full paths to variables in
#     # the variable nodes themselves.
#     # store relaive unpromoted paths in main model
#     # TODO: after flattening graph, attach full path to promoted name to
#     # all nodes use paths to promoted names as labels in networkx
