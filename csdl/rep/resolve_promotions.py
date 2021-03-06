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
from typing import List, Tuple, Dict, Set, Final, Literal, Union


def validate_promotions_and_split(
    qp: Dict[str, Set[str]],
    qup: Dict[str, Set[str]],
    local_namespace_source_shapes: Dict[str, Shape],
    local_namespace_target_shapes: Dict[str, Shape],
    namespace: str,
    model_type_name: str,
    promotes: Union[List[str], None],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    validate user specified promotions and store maps from promoted to
    unpromoted names; this function is never called on main model
    """
    # split dictionary between variables being promoted and not promoted
    if promotes is None:
        # promote all promotable variables
        return qp, dict()
    if len(promotes) == 0:
        return dict(), qup
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
    promoting: Dict[str, Set[str]] = dict(
        filter(lambda x: x[0] in promotes, qp.items()))
    not_promoting: Dict[str, Set[str]] = dict(
        filter(lambda x: x[0] not in promotes, qup.items()))
    return promoting, not_promoting


def resolve_promotions(
    model: 'Model',
    promotes: Union[List[str], None] = None,
    namespace: str = '',
) -> Tuple[Dict[str, Shape], Dict[str, Shape], Dict[
        str, Union[Input, Output]], Dict[str, DeclaredVariable]]:
    promoted_sources_from_children_shapes: Dict[str, Shape] = dict()
    promoted_targets_from_children_shapes: Dict[str, Shape] = dict()
    promoted_sources_from_children: Dict[str, Union[Input, Output]] = dict()
    promoted_targets_from_children: Dict[str, DeclaredVariable] = dict()

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
    locally_defined_sources: Final[Dict[str, Union[Input, Output]]] = c
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

    # update map from promoted to unpromoted names; variables that
    # are not promoted to this level will have the corresponding
    # child model's name prepended to their promoted path later on;
    # variables that are not promoted from this model to parent will
    # have name of current model prepended by parent

    # collect variables from children that will be promoted to parent
    # model and prepend namespace to promoted names of variables that
    # will not be promoted any further

    # () (x1)
    # (A) (x2)
    # (A.B) (x3)
    # (A.B.C) (x4)

    # promoting all
    # A.B.C pu: {x4: {x4}}
    # A.B pu: {x4: {C.x4}, x3: {x3}}
    # A pu: {x4: {B.C.x4}, x3: {B.x3}, x2: {x2}}
    # main pu: {x4: {A.B.C.x4}, x3: {A.B.x3}, x2: {A.x2}, x1: {x1}}

    # promoting none
    # A.B.C pu: {x4: {x4}}
    # A.B pu: {C.x4: {C.x4}, x3: {x3}}
    # A pu: {B.C.x4: {B.C.x4}, B.x3: {B.x3}, x1: {x1}}
    # main pu: {A.B.C.x4: {A.B.C.x4}, A.B.x3: {A.B.x3}, A.x2: {A.x2}, x1: {x1}}

    for s in model.subgraphs:
        # initialize pfc and npfc.
        # both pfc and npfc have keys corresponding to each variable in
        # child model. Therefore, len(pfc) = len(npfc) = n
        # eliminate keys in both pfc and npfc depending on
        # child model's promotions. At the end,
        # len(pfc) + len(npfc) = n

        # names as they appear in child namespace --> sets of unpromoted
        # names in this models' namespace
        # child model B:
        # pfc[x] --> {B.x, B.C.x}
        # pfc[C.y] --> {B.C.y, B.C.D.y}
        promoting_from_child: Dict[str, Set[str]] = {
            k: {prepend_namespace(s.name, vv)
                for vv in v}
            for k, v in s.submodel.promoted_to_unpromoted.items()
        }

        # names as they appear in this model's namespace prior to
        # promotion --> sets of unpromoted names in this models'
        # namespace
        # child model B:
        # pfc[B.x] --> {B.x, B.C.x}
        # pfc[B.C.y] --> {B.C.y, B.C.D.y}
        not_promoting_from_child: Dict[str, Set[str]] = {
            prepend_namespace(s.name, k):
            {prepend_namespace(s.name, vv)
             for vv in v}
            for k, v in s.submodel.promoted_to_unpromoted.items()
        }

        num_npfc = len(not_promoting_from_child)
        num_pfc = len(promoting_from_child)
        if num_npfc != num_pfc:
            raise ValueError('size mismatch')

        if s.promotes is None:
            # if promoting everything:
            # - npfc are all variables already with a namespace
            # - pfc are all variables without a namespace
            remove_keys = []
            for k in promoting_from_child.keys():
                if '.' in k:
                    remove_keys.append(k)
                else:
                    not_promoting_from_child.pop(
                        prepend_namespace(s.name, k))
            for k in remove_keys:
                promoting_from_child.pop(k)

        elif len(s.promotes) == 0:
            # if promoting nothing:
            # - npfc are all variables (leave as is)
            # - pfc is empty
            promoting_from_child = dict()
        else:
            # if promoting selectively:
            # - check that promotions are valid
            invalid_promotes = set(
                s.promotes) - (local_namespace_source_shapes.keys()
                               | local_namespace_target_shapes.keys())
            if invalid_promotes != set():
                raise KeyError(
                    "Invalid promotes {} specified in submodels within model {} of type {}"
                    .format(
                        invalid_promotes,
                        namespace,
                        type(model).__name__,
                    ))

            # filter promoted variables
            # remove_keys = []
            # for k in promoting_from_child.keys():
            #     if k not in s.promotes:
            #         remove_keys.append(k)
            # for k in remove_keys:
            #     promoting_from_child.pop(k)

            promoting_from_child = dict(
                filter(lambda x: x[0] in s.promotes,
                       promoting_from_child.items()))

            # filter unpromoted variables
            for k in promoting_from_child.keys():
                unpromoted_name = prepend_namespace(s.name, k)
                if unpromoted_name in not_promoting_from_child.keys():
                    not_promoting_from_child.pop(unpromoted_name)

        if len(promoting_from_child) + len(
                not_promoting_from_child) != num_pfc:
            raise ValueError('size mismatch')

        for k, v in promoting_from_child.items():
            # update set of unpromoted names for each promoted name
            if k in promoted_to_unpromoted.keys():
                promoted_to_unpromoted[k].update(v)
            else:
                promoted_to_unpromoted[k] = v
        for k, v in not_promoting_from_child.items():
            # update set of unpromoted names for each promoted name
            if k in promoted_to_unpromoted.keys():
                promoted_to_unpromoted[k].update(v)
            else:
                promoted_to_unpromoted[k] = v

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

    # gather sources and targets that are allowed to be promoted to
    # parent model prior to checking if shapes and values of variables
    # from sibling models and parent model are consistent.
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
