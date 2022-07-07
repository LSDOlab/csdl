from networkx import DiGraph
from csdl.rep.ir_node import IRNode
from csdl.rep.variable_node import VariableNode
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable
from typing import List


def merge_promoted_nodes(graph: DiGraph):
    """
    Assuming all nodes store their promoted names, merge sources and
    sinks that are connected because of promotions.
    The graph must be a flattened graph.
    Each node in the graph must contain its promoted (unique name).
    """
    nodes: List[IRNode] = list(graph.nodes())
    remove_nodes: List[IRNode] = []

    for a in nodes:
        for b in nodes:
            if isinstance(a, VariableNode) and isinstance(
                    b, VariableNode):
                # merge nodes that have the same promoted name
                if a.var.name == b.var.name:
                    if isinstance(a.var,
                                  (Input, Output)) and isinstance(
                                      b.var, DeclaredVariable):
                        graph.add_edges_from([
                            (a, s) for s in graph.successors(b)
                        ])
                        remove_nodes.append(b)
                    elif isinstance(b.var,
                                    (Input, Output)) and isinstance(
                                        a.var, DeclaredVariable):
                        graph.add_edges_from([
                            (b, s) for s in graph.successors(a)
                        ])
                        remove_nodes.append(b)
    graph.remove_nodes_from(remove_nodes)
