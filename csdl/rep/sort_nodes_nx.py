from typing import List
from csdl.utils.graph import modified_topological_sort_nx
from csdl.rep.ir_node import IRNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.model_node import ModelNode
from csdl.rep.get_registered_outputs_from_graph import get_registered_outputs_from_graph
from csdl.rep.get_model_nodes_from_graph import get_model_nodes_from_graph
from networkx import DiGraph, topological_sort

from csdl.rep.get_nodes import get_input_nodes, get_var_nodes
from csdl.utils.prepend_namespace import prepend_namespace


def sort_nodes_nx(
    graph: DiGraph,
    namespace: str = '',
    *,
    flat: bool,
) -> List[IRNode]:
    """
    Use Kahn's algorithm to sort nodes, reordering expressions without
    regard for the order in which user registers outputs

    **Returns**

    `List[Node]`
    : nodes in reverse order of execution that guarantees a triangular
    Jacobian structure
    """
    var_nodes = get_var_nodes(graph)
    input_nodes: List[VariableNode] = get_input_nodes(var_nodes)
    sorted_nodes: List[IRNode] = list(topological_sort(graph))
    sorted_nodes = list(set(input_nodes) -
                        set(sorted_nodes)) + sorted_nodes
    if flat is False:
        model_nodes: List[ModelNode] = get_model_nodes_from_graph(graph)
        for m in model_nodes:
            m.sorted_nodes = sort_nodes_nx(
                DiGraph(m.graph),
                namespace=m.namespace,
                flat=False,
            )
    return sorted_nodes
