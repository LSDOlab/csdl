from csdl.lang.apply_fn_to_submodels import apply_fn_to_submodels
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
from csdl.lang.model import Model
from csdl.lang.subgraph import Subgraph
from csdl.lang.output import Output
from csdl.utils.typehints import Shape
from typing import Callable, Dict, List, Tuple, Any, TypeVar, Set
from copy import copy
from networkx import DiGraph, simple_cycles

T = TypeVar('T', bound=Any)
U = TypeVar('U', bound=Any)

# declared variables no longer depend on subgraphs
# after promotions are evaluated, each output promoted from a subgraph
# will replace declared variable with same name and shape in parent
# model, and new output node will depend on subgraph
# note: if promotions are successful, output from subgraph and declared
# variable in parent model with same name will have same shape; no need
# to check twice


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
        model.promoted_targets.keys())


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
    graph = model.rep.unflat_graph
    return {
        x.name: x.shape
        for x in list(
            filter(
                lambda x: isinstance(x, Output) and x not in set(
                    model.registered_outputs),
                graph.nodes(),
            ))
    }
