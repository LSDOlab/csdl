from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
try:
    from csdl.lang.model import Model
except ImportError:
    pass
from csdl.lang.subgraph import Subgraph
from csdl.lang.output import Output
from csdl.utils.typehints import Shape
from csdl.utils.find_keys_with_matching_values import find_keys_with_matching_values
from typing import Callable, Dict, List, Tuple, Any, TypeVar, Set
from copy import copy
from networkx import DiGraph, simple_cycles

T = TypeVar('T', bound=Any)
U = TypeVar('U', bound=Any)


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


def find_names_with_matching_shapes(
    a: Dict[str, Shape],
    b: Dict[str, Shape],
) -> Set[str]:
    """
    Find names of variables whose shapes match. If any shapes mismatch,
    raise an error.
    """
    return find_keys_with_matching_values(
        a,
        b,
        error=lambda name1, name2, shape1, shape2: ValueError(
            "Shapes do not match; {} has shape {}, and {} has shape {}".
            format(name1, name2, shape1, shape2)))


def gather_variables_to_promote_no_futher(
        model: 'Model',
        locally_defined_promotable_names: Set[str]) -> Dict[str, Shape]:
    # verbose for strict type checking
    io: List[Input | Output | DeclaredVariable] = []
    io.extend(model.inputs)
    io.extend(model.registered_outputs)
    io.extend(model.declared_variables)
    variables_to_promote_no_further = {x.name: x.shape for x in io}
    for k in locally_defined_promotable_names:
        variables_to_promote_no_further.pop(k)
    return variables_to_promote_no_further


def collect_promoted_variable_names(
    model: 'Model',
    main: bool = True,
    promotes: List[str] | None = None,
) -> Tuple[Dict[str, Shape], Dict[str, Shape], Dict[str, Set[str]]]:
    """
    Collect variables promoted from submodels to `model` and return
    promotions specified by `promotes`.
    Updates `model.sources_promoted_from_submodels` and
    `model.sinks_promoted_from_submodels` so that CSDL can process
    connections later.
    """
    sources_promoted_from_submodels: Dict[str, Shape] = dict()
    sinks_promoted_from_submodels: Dict[str, Shape] = dict()

    for s in model.subgraphs:
        m = s.submodel
        p1, p2, pu = collect_promoted_variable_names(m, main=False)

        # required for checking if user has specified promotions that
        # result in cycles between models
        model.promoted_to_unpromoted = prepend_namespaces(s, pu)

        # Rule: each source (input or output) name promoted to this
        # level must be unique
        duplicate_source_names = sources_promoted_from_submodels.keys(
        ) & p1.keys()
        if duplicate_source_names != set():
            raise KeyError(
                "Cannot promote two inputs, two outputs, or an input and an output with the same name. Please check the variables with the following promoted paths: {}"
                .format(list(duplicate_source_names)))
        sources_promoted_from_submodels.update(p1)

        # Rule: all sinks (declared variable) with a common name
        # promoted to this level must have the same shape
        _ = find_names_with_matching_shapes(
            sinks_promoted_from_submodels, p2)
        sinks_promoted_from_submodels.update(p2)

    # Rule: if a source and a sink promoted to this level have matching
    # name and shape, then they form a connection
    connected_sinks = find_names_with_matching_shapes(
        sources_promoted_from_submodels, sinks_promoted_from_submodels)

    # Rule: If a declared variable is connected automatically by
    # promotion, then any further promotion will be applied to the
    # source (input or output) connected to the sink
    for c in connected_sinks:
        sinks_promoted_from_submodels.pop(c)

    # Rule: each source (input or output) name specified in this level
    # must not match the name of any source promoted to this level
    # verbose for strict type checking
    io: List[Input | Output] = []
    io.extend(model.inputs)
    io.extend(model.registered_outputs)
    locally_defined_sources: Dict[str, Shape] = {
        x.name: x.shape
        for x in io
    }
    locally_defined_promotable_sources: Dict[str, Shape] = dict(
        filter(
            lambda x: x[0] in set(promotes)
            if promotes is not None else True,
            locally_defined_sources.items()))
    duplicate_source_names = sources_promoted_from_submodels.keys(
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
        sinks_promoted_from_submodels,
        locally_defined_sinks,
    )
    print('sources_promoted_from_submodels',
          sources_promoted_from_submodels.keys())
    print('locally_defined_promotable_sources',
          locally_defined_promotable_sources.keys())

    sources_to_promote_to_parent = dict(
        sources_promoted_from_submodels,
        **locally_defined_promotable_sources,
    )
    sinks_to_promote_to_parent = dict(
        sinks_promoted_from_submodels,
        **locally_defined_promotable_sinks,
    )

    # store a copy of promotions in model for parent models to access,
    # but not modify unless specifically accessing this model's
    # attributes to ensure promoted variable names are unique
    variables_to_promote_no_further = gather_variables_to_promote_no_futher(
        model, locally_defined_promotable_names)
    model.sources_promoted_from_submodels = copy(
        sources_promoted_from_submodels).update(
            variables_to_promote_no_further)
    model.sinks_promoted_from_submodels = copy(
        sinks_promoted_from_submodels).update(
            variables_to_promote_no_further)

    # get the variables that will be promoted no further for parent
    # model to update its map from promoted paths to unpromoted paths
    for k in variables_to_promote_no_further.keys():
        model.promoted_to_unpromoted[k] = {k}

    # all models gather promoted names form submodels and hand them off
    # to their respective parent model, but the main model has no parent
    # model
    if main is True:
        model.promoted_sources.update(
            locally_defined_promotable_sources)
        model.promoted_sinks.update(locally_defined_promotable_sinks)

    model.sources_to_promote_to_parent = sources_to_promote_to_parent
    model.sinks_to_promote_to_parent = sinks_to_promote_to_parent

    return sources_to_promote_to_parent, sinks_to_promote_to_parent, model.promoted_to_unpromoted
