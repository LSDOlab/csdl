try:
    from csdl import Model
except ImportError:
    pass
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable
from csdl.rep.collect_locally_defined_variables import collect_locally_defined_variables
from csdl.rep.construct_unflat_graph import construct_unflat_graph
from csdl.utils.typehints import Shape
from csdl.utils.check_duplicate_keys import check_duplicate_keys
from csdl.utils.find_names_with_matching_shapes import find_names_with_matching_shapes
from csdl.utils.prepend_namespace import prepend_namespace
from networkx import topological_sort, simple_cycles
from typing import List, Tuple, Dict, Set, Final, Literal


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


def validate_promotions_and_split(
    q: Dict[str, Set[str]],
    local_namespace_source_shapes: Dict[str, Shape],
    local_namespace_target_shapes: Dict[str, Shape],
    namespace: str,
    model_type_name: str,
    promotes: list[str] | None,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    validate user specified promotions and store maps from promoted to
    unpromoted names; this function is never called on main model
    """

    if promotes is None:
        # split dictionary between variables being promoted and not
        # promoted; in this case, promote all promotable variables
        return q, dict()
    if len(promotes) == 0:
        return dict(), q
    else:
        # check that user has not specified invalid promotes
        invalid_promotes = set(promotes) - (
            local_namespace_source_shapes.keys()
            | local_namespace_target_shapes.keys())
        if invalid_promotes != set():
            raise KeyError(
                "Invalid promotes {} specified in submodels within model {} of type {}"
                .format(
                    invalid_promotes,
                    namespace,
                    model_type_name,
                ))

        # split dictionary between variables beting promoted and not
        # promoted
        promoting = dict(filter(lambda x: x[0] in promotes, q.items()))
        not_promoting = dict(
            filter(lambda x: x[0] not in promotes, q.items()))
        return promoting, not_promoting


def resolve_promotions(
    model: 'Model',
    promotes: List[str] | None = None,
    namespace: str = '',
) -> Tuple[Dict[str, Shape], Dict[str, Shape], Dict[
        str, Input | Output], Dict[str, DeclaredVariable]]:
    promoted_sources_from_children_shapes: Dict[str, Shape] = dict()
    promoted_targets_from_children_shapes: Dict[str, Shape] = dict()
    promoted_sources_from_children: Dict[str, Input | Output] = dict()
    promoted_targets_from_children: Dict[str, DeclaredVariable] = dict()
    # promoted_to_unpromoted_descendant_variables: Dict[
    # str, Set[str]] = dict()

    for s in model.subgraphs:
        m = s.submodel
        # promotions must be resolved from the bottom of the hierarchy
        # to the top, so start with recursion
        (
            promoted_sources_from_child_shapes,
            promoted_targets_from_child_shapes,
            promoted_sources_from_child,
            promoted_targets_from_child,
        ) = resolve_promotions(
            m,
            s.promotes,
            namespace=prepend_namespace(namespace, s.name),
        )

        # Rule: each source (input or output) name promoted to this
        # level must be unique; comparing variables promoted from child
        # models, not yet comparing variables to locally defined
        # variables
        check_duplicate_keys(
            promoted_sources_from_child_shapes,
            promoted_sources_from_children_shapes,
        )

        # Rule: all sinks (declared variable) with a common name
        # promoted to this level must have the same shape; comparing
        # variables promoted from child models, not yet comparing
        # variables to locally defined variables
        # TODO: check that values are the same
        _ = find_names_with_matching_shapes(
            promoted_targets_from_child_shapes,
            promoted_targets_from_children_shapes,
        )

        # checks pass, update name-shape pairs for variables promoted
        # from children; if these variables are promoted to parent
        # model, the parent model will need this information to check
        # that the promotions are valid
        promoted_sources_from_children_shapes.update(
            promoted_sources_from_child_shapes)
        promoted_targets_from_children_shapes.update(
            promoted_targets_from_child_shapes)

        # checks pass, update name-Variable pairs for variables promoted
        # from children; we use these containers to establish
        # model-variable and variable-model dependency relationships;
        # dependency relationships determine execution order and must
        # not contain any cycles between variables
        promoted_sources_from_children.update(
            promoted_sources_from_child)
        promoted_targets_from_children.update(
            promoted_targets_from_child)

        # update map from promoted to unpromoted names; variables that
        # are not promoted to this level will have the corresponding
        # child model's name prepended to their promoted path later on;
        # variables that are not promoted from this model to parent will
        #  have name of current model prepended by parent
        # for promoted_name, unpromoted_names in m.promoted_to_unpromoted.items(
        # ):
        #     new_unpromoted_names = {
        #         prepend_namespace(s.name, unpromoted_name)
        #         for unpromoted_name in unpromoted_names
        #     }
        #     try:
        #         promoted_to_unpromoted_descendant_variables[
        #             promoted_name].update(new_unpromoted_names)
        #     except:
        #         promoted_to_unpromoted_descendant_variables[
        #             promoted_name] = new_unpromoted_names

    # locally defined variable information is necessary for checking
    # that the remaining rules for promotion are followed and to send
    # information about locally defined variables that are candidates
    # for promotion further up the model hierarchy
    # the keys are variable names without '.' because they are defined
    # in the local namespace
    # sources are both inputs and outputs
    # targets are declared variables
    a, b, c, d = collect_locally_defined_variables(model)
    locally_defined_source_shapes: Final[Dict[str, Shape]] = a
    locally_defined_target_shapes: Final[Dict[str, Shape]] = b

    # also store variable objects themselves to establish dependency
    # relationships between variables and models in parent model
    locally_defined_sources: Final[Dict[str, Input | Output]] = c
    locally_defined_targets: Final[Dict[str, DeclaredVariable]] = d

    # Rule: each source (input or output) name specified in this level
    # must not match the name of any source promoted to this level from
    # children; comparing variables in child with variables in parent
    if len(promoted_sources_from_children_shapes) > 0:
        check_duplicate_keys(
            locally_defined_source_shapes,
            promoted_sources_from_children_shapes,
        )

    # Rule: all sinks (declared variable) with a common name specified
    # in this level must have the same shape as any sink promoted to
    # this level with the same name; comparing variables in child with
    # variables in parent
    # TODO: check that values are the same
    if len(promoted_targets_from_children_shapes) > 0:
        _ = find_names_with_matching_shapes(
            promoted_targets_from_children_shapes,
            locally_defined_target_shapes,
        )

    # Rule: if a source and a target with the same name must have the
    # same shape in order to be promoted to the same model
    _ = find_names_with_matching_shapes(
        locally_defined_source_shapes,
        promoted_targets_from_children_shapes,
    )
    _ = find_names_with_matching_shapes(
        promoted_sources_from_children_shapes,
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

    # promoted_to_unpromoted is a dictionary.
    # The keys are strings.
    # The values are sets of strings.
    # Each key is the promoted name in the current model's namespace of
    # a variable.
    # Each value is a set of unpromoted names of variables promoted to
    # the current model.
    # The number of keys is the same as the number of variables in the
    # model hierarchy with the current model at the top of the
    # hierarchy.
    # The number of keys is the same as the number of all variables if
    # the current model is the model at the top of the hierarchy, which
    # contains all other models.
    # Sets of strings are stored as the dictionary values because a
    # variable promoted to one model may have been declared/created in
    # multiple submodels.
    # For example, an input/output from one model could be promoted to
    # the same model as a declared variable from another model is.
    # These two variables have different unpromoted names, but they have
    # the same promoted name.
    promoted_to_unpromoted: Dict[str, Set[str]] = dict()

    # store map from promoted to unpromoted names to enable issuing
    # connections using promoted or unpromoted names; all names are
    # relative to this model; map will be used to inform parent model of
    # promoted names and corresponding unpromoted names
    for k in locally_defined_source_shapes.keys():
        promoted_to_unpromoted[k] = {k}
    for k in locally_defined_target_shapes.keys():
        promoted_to_unpromoted[k] = {k}

    # collect variables from children that will be promoted to parent
    # model and prepend namespace to promoted names of variables that
    # will not be promoted any further
    for s in model.subgraphs:
        q = {
            k: {prepend_namespace(s.name, vv)
                for vv in v}
            for k, v in s.submodel.promoted_to_unpromoted.items()
        }
        promoting_from_child = dict()
        not_promoting_from_child = dict()
        if s.promotes is None:
            # split dictionary between variables being promoted and not
            # promoted; in this case, promote all promotable
            # variables
            promoting_from_child = q
        else:
            if len(s.promotes) == 0:
                not_promoting_from_child = q
            else:
                # check that user has not specified invalid promotes
                invalid_promotes = set(s.promotes) - (
                    local_namespace_source_shapes.keys()
                    | local_namespace_target_shapes.keys())
                if invalid_promotes != set():
                    raise KeyError(
                        "Invalid promotes {} specified in submodels within model {} of type {}"
                        .format(
                            invalid_promotes,
                            namespace,
                            type(model).__name__,
                        ))

                # split dictionary between variables being promoted and
                # not promoted
                promoting_from_child = dict(
                    filter(lambda x: x[0] in s.promotes, q.items()))
                not_promoting_from_child = dict(
                    filter(lambda x: x[0] not in s.promotes, q.items()))

        for k, v in promoting_from_child.items():
            # update set of unpromoted names for each promoted name
            try:
                promoted_to_unpromoted[k].update(v)
            except:
                promoted_to_unpromoted[k] = v
        for k, v in not_promoting_from_child.items():
            # update set of unpromoted names for each promoted name
            # prepending namespace to promoted names
            promoted_to_unpromoted[prepend_namespace(s.name, k)] = v

    model.promoted_to_unpromoted = promoted_to_unpromoted

    # store promoted sources and targets separately to look up when
    # issuing connections
    model.promoted_source_shapes.update(
        promoted_sources_from_children_shapes)
    model.promoted_source_shapes.update(locally_defined_source_shapes)
    model.promoted_target_shapes.update(
        promoted_targets_from_children_shapes)
    model.promoted_target_shapes.update(locally_defined_target_shapes)

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
