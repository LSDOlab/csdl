from typing import List
from csdl.utils.graph import modified_topological_sort_nx
from csdl.rep.ir_node import IRNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.model_node import ModelNode
from csdl.rep.get_registered_outputs_from_graph import get_registered_outputs_from_graph
from csdl.rep.get_model_nodes_from_graph import get_model_nodes_from_graph
from networkx import DiGraph, topological_sort


def sort_nodes_nx_old(
    graph: DiGraph,
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
    registered_outputs: list[
        VariableNode] = get_registered_outputs_from_graph(graph)
    if flat is True:
        sorted_nodes: List[IRNode] = modified_topological_sort_nx(
            graph,
            registered_outputs,
        )
    else:
        model_nodes: list[ModelNode] = get_model_nodes_from_graph(graph)
        sorted_nodes: List[IRNode] = modified_topological_sort_nx(
            graph,
            registered_outputs,
        )
        for m in model_nodes:
            m.sorted_nodes = sort_nodes_nx(
                DiGraph(m.graph),
                flat=False,
            )
    return sorted_nodes


from csdl.rep.get_inputs_from_graph import get_inputs_from_graph


def sort_nodes_nx(
    graph: DiGraph,
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
    input_nodes: List[VariableNode] = get_inputs_from_graph(graph)
    print([(x.name, y.name) for (x, y) in graph.edges()])
    sorted_nodes: List[IRNode] = list(topological_sort(graph))
    sorted_nodes = list(set(input_nodes) -
                        set(sorted_nodes)) + sorted_nodes
    if flat is False:
        model_nodes: list[ModelNode] = get_model_nodes_from_graph(graph)
        for m in model_nodes:
            m.sorted_nodes = sort_nodes_nx(
                DiGraph(m.graph),
                flat=False,
            )
    return sorted_nodes
