from ast import Call
from sre_constants import RANGE_UNI_IGNORE
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
from typing import Callable, Dict, Iterable, List, Tuple, Any, TypeVar, Set
from copy import copy
from networkx import DiGraph, simple_cycles
from csdl.utils.check_duplicate_keys import check_duplicate_keys
from csdl.utils.find_names_with_matching_shapes import find_names_with_matching_shapes

T = TypeVar('T', bound=Any)
U = TypeVar('U', bound=Any)


def prepend_namespaces_from_descendants(
    name: str,
    promoted_to_unpromoted: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """
    Prepend namespaces to each promoted name (key) and each unpromoted
    name in each value.
    Use for prepending namespaces to names of variables promoted from
    submodels and their descendants, but not promoted to current model.
    """
    promoted_to_unpromoted2: Dict[str, Set[str]] = dict()
    for k, v in promoted_to_unpromoted.items():
        promoted_to_unpromoted2[name + '.' +
                                k] = {name + '.' + x
                                      for x in v}
    return promoted_to_unpromoted2


def prepend_namespaces_this_level(
    name: str,
    promoted_to_unpromoted: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    """
    Prepend namespaces to each promoted name (key) and each unpromoted
    name in each value.
    Use for prepending namespaces to names of variables promoted to
    current model.
    """
    promoted_to_unpromoted2: Dict[str, Set[str]] = dict()
    for k, v in promoted_to_unpromoted.items():
        promoted_to_unpromoted2[k] = {name + '.' + x for x in v}
    return promoted_to_unpromoted2



def move_pairs_from_dict_to_dict(
    source: Dict[T, U],
    target: Dict[T, U],
    keys: Iterable[T],
):
    """
    Move key-value pair in `keys` from `source` to `target`.
    If `source` does not contain a key in `keys`, an error will be
    raised.
    If `target` already contains a key in `keys`, an error will be
    raised.
    """
    for k in keys:
        if k in target.keys():
            raise KeyError(
                "Key {} in source dictionary already exists in target dictionary"
                .format(k))
        target[k] = source.pop(k)



def add_promotable_names(
    promoted_to_unpromoted: Dict[str, Set[str]],
    promotable_variables: Dict[str, Shape],
):
    """
    add promotable names to map from promoted to unpromoted names
    """
    for k in promotable_variables.keys():
        try:
            promoted_to_unpromoted[k].add(k)
        except:
            promoted_to_unpromoted[k] = {k}


def collect_promoted_variable_names(
    model: 'Model',
    main: bool = True,
    promotes:Union[List[str], None] = None,
) -> Tuple[Dict[str, Shape], Dict[str, Shape], Dict[str, Set[str]],
           Dict[str, Set[str]]]:
    """
    Collect variables promoted from submodels to `model` and return
    promotions specified by `promotes`.
    Updates `model.sources_promoted_from_submodels` and
    `model.sinks_promoted_from_submodels` so that CSDL can process
    connections later.
    """
    sources_promoted_from_submodels: Dict[str, Shape] = dict()
    sinks_promoted_from_submodels: Dict[str, Shape] = dict()
    # dictionary of promoted to unpromoted names for all variables
    # promoted to descendant submodels, but no further
    promoted_to_unpromoted_promoted_to_this_level: Dict[
        str, Set[str]] = dict()
    # dictionary of promoted to unpromoted names for all variables
    # promoted to this model, and not to descendant submodels
    promoted_to_unpromoted_promote_to_parent: Dict[str,
                                                   Set[str]] = dict()
    for s in model.subgraphs:
        m = s.submodel
        submodel_name = s.name
        sources, sinks, pu, pu_descendants = collect_promoted_variable_names(
            m, main=False)
        print("In model {}, we have sources {} and sinks {}".format(
            submodel_name, sources.keys(), sinks.keys()))

        # Rule: each source (input or output) name promoted to this
        # level must be unique
        check_duplicate_keys(sources, sources_promoted_from_submodels)
        sources_promoted_from_submodels.update(sources)

        # Rule: all sinks (declared variable) with a common name
        # promoted to this level must have the same shape
        _ = find_names_with_matching_shapes(
            sinks_promoted_from_submodels, sinks)
        sinks_promoted_from_submodels.update(sinks)

        # prepend namespace to promoted and unpromoted names for all
        # variables promoted to a lower level/namespace, but not this
        # level/namespace
        promoted_to_unpromoted_promoted_to_this_level.update(
            prepend_namespaces_from_descendants(submodel_name,
                                                pu_descendants))

        # prepend namespace to unpromoted names for all variables that
        # CAN be promoted to this level/namespace; will move variables
        # that user DOES NOT promote out of this container
        promoted_to_unpromoted_promote_to_parent.update(
            prepend_namespaces_this_level(submodel_name, pu))

        print(
            "We also have variables promoted to model {}, but no further: {}"
            .format(submodel_name, pu_descendants.items()))
        print("We also have variables promoted to model {}: {}".format(
            submodel_name, pu))

    # Rule: if a source and a sink promoted to this level have matching
    # name and shape, then they form a connection
    connected_sinks = find_names_with_matching_shapes(
        sources_promoted_from_submodels, sinks_promoted_from_submodels)

    # Rule: If a declared variable is connected automatically by
    # promotion, then any further promotion will be applied to the
    # source (input or output) connected to the sink (declared
    # variable), not the sink
    for c in connected_sinks:
        sinks_promoted_from_submodels.pop(c)

    # verbose for strict type checking
    io: List[Union[Input, Output]] = []
    io.extend(model.inputs)
    io.extend(model.registered_outputs)
    locally_defined_sources: Dict[str, Shape] = {
        x.name: x.shape
        for x in io
    }

    # Rule: each source (input or output) name specified in this level
    # must not match the name of any source promoted to this level from
    # children
    check_duplicate_keys(
        locally_defined_sources,
        sources_promoted_from_submodels,
    )

    locally_defined_sinks: Dict[str, Shape] = {
        x.name: x.shape
        for x in model.declared_variables
    }

    # Rule: all sinks (declared variable) with a common name specified
    # in this level must have the same shape as any sink promoted to
    # this level with the same name
    _ = find_names_with_matching_shapes(
        sinks_promoted_from_submodels,
        locally_defined_sinks,
    )

    # ensure user has specified valid promotions only
    promotable_names: Set[str] = locally_defined_sources.keys(
    ) | locally_defined_sinks.keys(
    ) | sources_promoted_from_submodels.keys(
    ) | sinks_promoted_from_submodels.keys()
    invalid_promotes: Set[str] = set(
        promotes) - promotable_names if promotes is not None else set()
    if invalid_promotes != set():
        # TODO: include model _csdl_ name in error message
        raise KeyError(
            # "The following variables are not valid for promotion in model with {} {} of type {}: {}".format(
            # "autogenerated name" if autogenerated_name is True
            # else "user-specified name", name,
            "The following variables are not valid for promotion because they are neither defined in model of type {}: {}, nor promoted from models in lower levels of the model hiearchy"
            .format(type(model).__name__),
            invalid_promotes)

    # gather variables from this model that user has specified to
    # promote to parent model
    locally_defined_promotable_sources: Dict[str, Shape] = dict(
        filter(
            lambda x: x[0] in set(promotes)
            if promotes is not None else True,
            locally_defined_sources.items()))
    locally_defined_promotable_sinks: Dict[str, Shape] = dict(
        filter(
            lambda x: x[0] in set(promotes)
            if promotes is not None else True,
            locally_defined_sinks.items()))

    print('sources_promoted_from_submodels',
          sources_promoted_from_submodels.keys())
    print('locally_defined_promotable_sources',
          locally_defined_promotable_sources.keys())

    # KLUDGE: these steps could probably be simplified or combined
    add_promotable_names(
        promoted_to_unpromoted_promote_to_parent,
        locally_defined_promotable_sources,
    )
    add_promotable_names(
        promoted_to_unpromoted_promote_to_parent,
        locally_defined_promotable_sinks,
    )
    variables_to_promote_no_further = promotable_names - set(
        promotes) if promotes is not None else set()
    move_pairs_from_dict_to_dict(
        promoted_to_unpromoted_promote_to_parent,
        promoted_to_unpromoted_promoted_to_this_level,
        variables_to_promote_no_further,
    )

    sources_to_promote_to_parent = dict(
        {
            k: sources_promoted_from_submodels[k]
            for k in list(
                filter(
                    lambda x: x not in variables_to_promote_no_further,
                    sources_promoted_from_submodels.keys()))
        },
        **locally_defined_promotable_sources,
    )
    sinks_to_promote_to_parent = dict(
        {
            k: sinks_promoted_from_submodels[k]
            for k in list(
                filter(
                    lambda x: x not in variables_to_promote_no_further,
                    sinks_promoted_from_submodels.keys()))
        },
        **locally_defined_promotable_sinks,
    )

    # store a copy of promotions in model for parent models to access,
    # but not modify unless specifically accessing this model's
    # attributes to ensure promoted variable names are unique
    # model.sources_promoted_from_submodels = sources_promoted_from_submodels
    # model.sinks_promoted_from_submodels =
    # sinks_promoted_from_submodels
    # TODO: describe better; is this for connections??
    model.promoted_to_unpromoted = dict(
        promoted_to_unpromoted_promote_to_parent,
        **promoted_to_unpromoted_promoted_to_this_level,
    )

    # all models gather promoted names form submodels and hand them off
    # to their respective parent model, but the main model has no parent
    # model, so we store the promoted names in the main model
    if main is True:
        # model.promoted_sources.update(
        #     locally_defined_promotable_sources)
        # model.promoted_sinks.update(locally_defined_promotable_sinks)
        for k in locally_defined_sources:
            model.promoted_to_unpromoted[k] = {k}
        for k in locally_defined_sinks:
            model.promoted_to_unpromoted[k] = {k}

    model.sources_to_promote_to_parent = sources_to_promote_to_parent
    model.sinks_to_promote_to_parent = sinks_to_promote_to_parent

    return sources_to_promote_to_parent, sinks_to_promote_to_parent, promoted_to_unpromoted_promote_to_parent, promoted_to_unpromoted_promoted_to_this_level
