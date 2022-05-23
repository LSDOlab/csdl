try:
    from csdl.rep.graph_representation import GraphRepresentation
except ImportError:
    pass
from csdl.rep.get_implicit_operations_from_graph import get_implicit_operations_from_graph
from csdl.rep.operation_node import OperationNode
from csdl.lang.implicit_operation import ImplicitOperation
from csdl.lang.bracketed_search_operation import BracketedSearchOperation
from typing import Callable


def apply_fn_to_implicit_operation_nodes(
    rep: 'GraphRepresentation',
    fn: Callable,
):
    implicit_operation_nodes: list[
        OperationNode] = get_implicit_operations_from_graph(
            rep.flat_graph)
    for implicit in implicit_operation_nodes:
        if isinstance(implicit.op,
                      (ImplicitOperation, BracketedSearchOperation)):
            implicit.op._model.rep = GraphRepresentation(
                implicit.op._model)
            apply_fn_to_implicit_operation_nodes(
                implicit.op._model.rep,
                fn,
            )