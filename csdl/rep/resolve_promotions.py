try:
    from csdl import Model
except ImportError:
    pass
from curses import KEY_MARK
from unicodedata import name
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable
from csdl.rep.collect_locally_defined_variables import collect_locally_defined_variables
from csdl.rep.construct_unflat_graph import construct_unflat_graph
from csdl.utils.typehints import Shape
from csdl.utils.check_duplicate_keys import check_duplicate_keys
from csdl.utils.find_names_with_matching_shapes import find_names_with_matching_shapes
from networkx import DiGraph, topological_sort, simple_cycles
from typing import List, Tuple, Dict, Set, Final, Literal
from copy import copy, deepcopy


def check_for_cycles(
    model: 'Model',
    model_path: str,
    mode: Literal['promotions', 'connections'],
):
    # build a graph of the model at this level to detect cycles created
    # between models by promoting variables; note: this does not
    # necessarily mean thet there is a cyclic relationship between
    # variables at different levels in the model hierarchy or across
    # model boundaries at the same level of the model hierarchy; these
    # cycles may be due to unnecessary feedbacks across model boundaries
    graph = construct_unflat_graph(
        model.inputs,
        model.registered_outputs,
        model.subgraphs,
        recursive=False,
    )

    try:
        _ = topological_sort(graph)
    except:
        model.model_cycles = [
            x.name for x in list(simple_cycles(graph))
        ]
        if mode == 'promotions':
            raise Warning(
                "User specified promotions created the following cycles within model named {} of type {}: {}. If using a CSDL back end/code generator that uses the flattened graph representation of a model and no errors are raised, you may safely disregard this message. Otherwise, these cycles introduce unnecessary feedback in your model and will affect performance."
                .format(model_path,
                        type(model).__name__),
                model.model_cycles,
            )
        elif mode == 'connections':
            pass


def split_promoting_not_promoting(
    promoted_to_unpromoted_descendant_variables: Dict[str, Set[str]],
    local_namespace_source_shapes: Dict[str, Shape],
    local_namespace_target_shapes: Dict[str, Shape],
    model_path: str,
    model_type_name: str,
    promotes: list[str] | None,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    validate user specified promotions and store maps from promoted to
    unpromoted names; this function is never called on main model
    """
    if promotes is not None:
        # check that user has not specified invalid promotes
        invalid_promotes = set(promotes) - (
            local_namespace_source_shapes.keys()
            | local_namespace_target_shapes.keys())
        if invalid_promotes != set():
            raise KeyError(
                "Invalid promotes {} specified in submodels within model {} of type {}"
                .format(
                    invalid_promotes,
                    model_path,
                    model_type_name,
                ))

        # split dictionary between variables beting promoted and not
        # promoted
        promoting = dict(
            filter(lambda kv: kv[0] in promotes,
                   promoted_to_unpromoted_descendant_variables.items()))
        not_promoting = dict(
            filter(lambda kv: kv[0] not in promotes,
                   promoted_to_unpromoted_descendant_variables.items()))
        # prepend namespace of submodel to each variable name that is
        # not promoted
        # TODO: verify that this condition is always true since this
        # function is never called on main model
        if model_path != '':
            not_promoting = {
                model_path.rsplit('.')[-1] + k:
                {model_path.rsplit('.')[-1] + kk
                 for kk in v}
                for k, v in not_promoting.items()
            }
        return promoting, not_promoting
    else:
        # split dictionary between variables beting promoted and not
        # promoted; in this case, promote all promotable variables
        return promoted_to_unpromoted_descendant_variables, dict()


def resolve_promotions(
    model: 'Model',
    promotes: List[str] | None = None,
    model_path: str = '',
) -> Tuple[Dict[str, Shape], Dict[str, Shape], Dict[
        str, Input | Output], Dict[str, DeclaredVariable]]:
    promoted_sources_from_children_shapes: Dict[str, Shape] = dict()
    promoted_targets_from_children_shapes: Dict[str, Shape] = dict()
    promoted_sources_from_children: Dict[str, Input | Output] = dict()
    promoted_targets_from_children: Dict[str, DeclaredVariable] = dict()
    promoted_to_unpromoted_descendant_variables: Dict[
        str, Set[str]] = dict()

    for s in model.subgraphs:
        m = s.submodel
        # promotions must be resolved from the bottom of the hierarchy
        # to the top, so start with recursion
        a, b, c, d = resolve_promotions(
            m,
            s.promotes,
            model_path=s.name if model_path == '' else model_path +
            '.' + s.name,
        )

        # Rule: each source (input or output) name promoted to this
        # level must be unique
        check_duplicate_keys(
            a,
            promoted_sources_from_children_shapes,
        )

        # Rule: all sinks (declared variable) with a common name
        # promoted to this level must have the same shape
        _ = find_names_with_matching_shapes(
            b,
            promoted_targets_from_children_shapes,
        )

        # checks pass, update name-shape pairs for variables promoted
        # from children; if these variables are promoted to parent
        # model, the parent model will need this information to check
        # that the promotions are valid
        promoted_sources_from_children_shapes.update(a)
        promoted_targets_from_children_shapes.update(b)

        # checks pass, update name-Variable pairs for variables promoted
        # from children; we use these containers to establish
        # model-variable and variable-model dependency relationships;
        # dependency relationships determine execution order and must
        # not contain any cycles between variables
        promoted_sources_from_children.update(c)
        promoted_targets_from_children.update(d)

        # update map from promoted to unpromoted names; variables that
        # are not promoted to this level will have the corresponding
        # child model's name prepended to their promoted path later on;
        # variables that are not promoted to parent will have name of
        # current model prepended later on
        promoted_to_unpromoted_descendant_variables.update({
            k: {kk if s.name == '' else s.name + '.' + kk
                for kk in v}
            for k, v in m.promoted_to_unpromoted.items()
        })

    # locally defined variable information is necessary for checking
    # that the remaining rules for promotion are followed and to send
    # information about locally defined variables that are candidates
    # for promotion further up the model hierarchy
    a, b, c, d = collect_locally_defined_variables(model)
    locally_defined_source_shapes: Final[Dict[str, Shape]] = a
    locally_defined_target_shapes: Final[Dict[str, Shape]] = b
    locally_defined_sources: Final[Dict[str, Input | Output]] = c
    locally_defined_targets: Final[Dict[str, DeclaredVariable]] = d

    # Rule: each source (input or output) name specified in this level
    # must not match the name of any source promoted to this level from
    # children
    if len(promoted_sources_from_children_shapes) > 0:
        check_duplicate_keys(
            locally_defined_source_shapes,
            promoted_sources_from_children_shapes,
        )

    # Rule: all sinks (declared variable) with a common name specified
    # in this level must have the same shape as any sink promoted to
    # this level with the same name
    if len(promoted_targets_from_children_shapes) > 0:
        _ = find_names_with_matching_shapes(
            promoted_targets_from_children_shapes,
            locally_defined_target_shapes,
        )

    # these containers are used later for validating promotions and
    # informing parent model of which variables are valid options for
    # users to specify promotions
    local_namespace_source_shapes: Final[Dict[str, Shape]] = dict(
        locally_defined_source_shapes,
        **promoted_sources_from_children_shapes,
    ) if len(promoted_sources_from_children_shapes
             ) > 0 else locally_defined_source_shapes
    local_namespace_target_shapes: Final[Dict[str, Shape]] = dict(
        locally_defined_target_shapes,
        **promoted_targets_from_children_shapes,
    ) if len(promoted_targets_from_children_shapes
             ) > 0 else locally_defined_target_shapes

    # store map from promoted to unpromoted names to enable issuing
    # connections using promoted or unpromoted names; all names are
    # relative to this model; map will be used to inform parent model of
    # promoted names and corresponding unpromoted names
    promoted_to_unpromoted: Dict[str, Set[str]] = dict()
    for k in locally_defined_source_shapes.keys():
        promoted_to_unpromoted[k] = {k}
    for k in locally_defined_target_shapes.keys():
        promoted_to_unpromoted[k] = {k}

    # collect variables from children that will be promoted to parent
    # model and prepend namespace to promoted names of variables that
    # will not be promoted any further
    for s in model.subgraphs:
        promoting_from_children, not_promoting_from_children = split_promoting_not_promoting(
            promoted_to_unpromoted_descendant_variables,
            local_namespace_source_shapes,
            local_namespace_target_shapes,
            model_path,
            type(model).__name__,
            s.promotes,
        )
        for k, v in promoting_from_children.items():
            # update set of unpromoted names for each promoted name
            try:
                promoted_to_unpromoted[k].update(v)
            except:
                promoted_to_unpromoted[k] = v
        for k, v in not_promoting_from_children.items():
            # update set of unpromoted names for each promoted name
            # prepending namespace to promoted names
            key = k if s.name == '' else s.name + '.' + k
            promoted_to_unpromoted[key] = v

    model.promoted_to_unpromoted = promoted_to_unpromoted
    print('model.promoted_to_unpromoted', model.promoted_to_unpromoted)

    # # store promoted sources and targets separately to look up when
    # # issuing connections
    # model.promoted_sources.update(
    #     set(promoted_sources_from_children_shapes.keys()))
    # model.promoted_sources.update(
    #     set(locally_defined_source_shapes.keys()))
    # model.promoted_targets.update(
    #     set(promoted_targets_from_children_shapes.keys()))
    # model.promoted_targets.update(
    #     set(locally_defined_target_shapes.keys()))

    # collect name-shape pairs for all variables that are available for
    # user to promote to parent model
    user_promotable_sources_shapes = local_namespace_source_shapes if promotes is None else dict(
        filter(
            lambda kv: kv[0] in set(promotes),
            local_namespace_source_shapes.items(),
        ))
    user_promotable_targets_shapes = local_namespace_target_shapes if promotes is None else dict(
        filter(
            lambda kv: kv[0] in set(promotes),
            local_namespace_target_shapes.items(),
        ))

    # add dependency relationship representing data flowing from local
    # source to child model
    for s in model.subgraphs:
        for local_name, src in locally_defined_sources.items():
            for lower_level_name, _ in promoted_targets_from_children.items(
            ):
                if local_name == lower_level_name:
                    s.add_dependency_node(src)
                    src.add_dependent_node(s)

    # add dependency relationship representing data flowing from child
    # model to locally declared variable
    for s in model.subgraphs:
        for local_name, tgt in locally_defined_targets.items():
            for lower_level_name, _ in promoted_sources_from_children.items(
            ):
                if local_name == lower_level_name:
                    tgt.add_dependency_node(s)
                    s.add_dependent_node(tgt)

    # add dependency relationship representing data flowing from locally
    # declared variable to child model that declares a variable
    # with same name and shape that has been promoted to this model
    for s in model.subgraphs:
        for local_name, tgt in locally_defined_targets.items():
            for lower_level_name, _ in promoted_targets_from_children.items(
            ):
                if local_name == lower_level_name:
                    s.add_dependency_node(tgt)
                    tgt.add_dependent_node(s)

    check_for_cycles(model, model_path, 'promotions')

    # collect Variable objects for parent model to establish dependency
    # relationships between Variable objects and Model objects
    sources = dict(locally_defined_sources,
                   **promoted_sources_from_children)
    promotable_sources = {
        k: sources[k]
        for k in user_promotable_sources_shapes.keys()
    }
    targets = dict(locally_defined_targets,
                   **promoted_targets_from_children)
    promotable_targets = {
        k: targets[k]
        for k in user_promotable_targets_shapes.keys()
    }
    return user_promotable_sources_shapes, user_promotable_targets_shapes, promotable_sources, promotable_targets
