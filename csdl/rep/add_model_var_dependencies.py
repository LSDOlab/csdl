try:
    from csdl.lang.model import Model
except ImportError:
    pass
try:
    from csdl.lang.input import Input
except ImportError:
    pass
try:
    from csdl.lang.output import Output
except ImportError:
    pass

from typing import Dict, List, Union
from csdl.utils.typehints import Shape
from csdl.lang.node import Node
from csdl.lang.subgraph import Subgraph


def detect_cycle(
    start: Subgraph,
    n: Node,
    namespace: Union[str, None] = None,
):
    for prev in start.dependencies:
        if prev is start:
            raise RecursionError(
                "Cycle detected due to model {} of type {}".format(
                    start.name if namespace is None else namespace +
                    '.' + start.name, type(start.submodel)), )
        detect_cycle(start, prev)


def add_model_var_dependencies_due_to_promotions(
    model: 'Model',
    namespace: Union[str, None] = None,
):
    """
    After resolving promotions, establish dependency relationships
    between models and variables in the parent model.
    This function will check if there are any resulting cycles between
    models and variables.
    """
    for s in model.subgraphs:
        m = s.submodel
        add_model_var_dependencies_due_to_promotions(m)
        promoted_sources: Dict[
            str, Shape] = m.sources_to_promote_to_parent
        promoted_sinks: Dict[
            str, Shape] = m.sinks_to_promote_to_parent

        # make each declared variable that is connected to submodel
        # source depend on model in graph
        for var in model.declared_variables:
            if var.name in promoted_sources.keys():
                if var.shape == promoted_sources[var.name]:
                    var.add_dependency_node(s)
                    s.add_dependent_node(var)
                    detect_cycle(
                        s,
                        var,
                        namespace=s.name if namespace is None else
                        namespace + '.' + s.name,
                    )

        # verbose for strict type checking
        io: List[Union[Input, Output]] = []
        io.extend(model.registered_outputs)
        io.extend(model.inputs)

        print('sources', [x.name for x in io])
        print('prmoted_sinks', promoted_sinks.keys())

        # make each declared variable in submodel that is connected to
        # model source depend on source in graph
        for var in io:
            if var.name in promoted_sinks.keys(
            ) and var.shape == promoted_sinks[var.name]:
                s.add_dependency_node(var)
                var.add_dependent_node(s)
                for d in s.dependencies:
                    detect_cycle(
                        s,
                        d,
                        namespace=s.name if namespace is None else
                        namespace + '.' + s.name,
                    )


# TODO: check that connections do not create cycles after promotions are
# resolved, but before nodes are sorted; i.e. we don't know how to order
# models until we check connections
def add_model_var_dependencies_due_to_connections(model: 'Model'):
    """
    After resolving connections, establish dependency relationships
    between models and variables in the parent model.
    This function will check if there are any resulting cycles between
    models and variables.
    """
    pass
