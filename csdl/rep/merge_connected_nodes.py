from networkx import DiGraph
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from typing import List, Tuple, Dict


def merge_connected_nodes(graph: DiGraph,
                          connections: List[Tuple[str, str]]):
    sink_to_source: Dict[str, str] = {b: a for (a, b) in connections}
    nodes: Dict[str, VariableNode] = {
        node.var.abs_prom_name: node
        for node in list(
            filter(lambda x: isinstance(x, VariableNode),
                   graph.nodes()))
    }
    sinks = []
    for k, v in nodes.items():
        if k in sink_to_source.keys():
            source = nodes[sink_to_source[k]]
            source.var.secondary_name = k
            successors: List[OperationNode] = list(graph.successors(v))
            edges = [(source, succ) for succ in successors]
            graph.add_edges_from(edges)
            sinks.append(v)
    graph.remove_nodes_from(sinks)
