from typing import List

from csdl.rep.graph_representation import GraphRepresentation
from csdl.rep.ir_node import IRNode
from csdl.rep.model_node import ModelNode
from csdl.rep.sort_nodes_nx import sort_nodes_nx
from csdl.rep.apply_fn_to_implicit_operation_nodes import apply_fn_to_implicit_operation_nodes
from csdl.rep.get_model_nodes_from_graph import get_model_nodes_from_graph
from networkx import DiGraph


def _remove_dead_code_nodes_from_graph(
    graph: DiGraph,
    sorted_nodes: List[IRNode],
    flat: bool = True,
):
    """
    After sorting nodes, remove nodes corresponding to dead code.
    Dead code is any code that is not upstream of registered outputs.
    """
    remove = []
    sn = set(sorted_nodes)
    for node in graph.nodes():
        if node not in sn:
            remove.append(node)
    graph.remove_nodes_from(remove)

    if flat is False:
        model_nodes: list[ModelNode] = get_model_nodes_from_graph(graph)
        for m in model_nodes:
            _remove_dead_code_nodes_from_graph(
                m.graph,
                m.sorted_nodes,
                flat=False,
            )


def remove_dead_code(rep: GraphRepresentation) -> GraphRepresentation:
    # sort remove dead code; requires sorting nodes
    rep.flat_sorted_nodes = sort_nodes_nx(
        rep.flat_graph,
        flat=True,
    )
    _remove_dead_code_nodes_from_graph(
        rep.flat_graph,
        rep.flat_sorted_nodes,
        flat=True,
    )

    rep.unflat_sorted_nodes = sort_nodes_nx(
        rep.unflat_graph,
        flat=False,
    )
    _remove_dead_code_nodes_from_graph(
        rep.unflat_graph,
        rep.unflat_sorted_nodes,
        flat=False,
    )

    apply_fn_to_implicit_operation_nodes(rep, remove_dead_code)
    return rep
