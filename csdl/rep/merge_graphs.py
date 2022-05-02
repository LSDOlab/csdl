try:
    from csdl.lang.model import Model
except ImportError:
    pass
try:
    from csdl.lang.subgraph import Subgraph
except ImportError:
    pass
from csdl.rep.model_node import ModelNode
from csdl.rep.ir_node import IRNode
from typing import List, Tuple
from networkx import DiGraph


def merge_graphs(graph: DiGraph):
    """
    Copy nodes and edges from graphs in submodels into the graph for the
    main model.
    """
    model_nodes: List[ModelNode] = list(
        filter(lambda x: isinstance(x, ModelNode), graph.nodes()))
    nodes: List[Tuple[IRNode, IRNode]] = []
    edges: List[Tuple[IRNode, IRNode]] = []
    if len(model_nodes) == 0:
        return graph
    for mn in model_nodes:
        g = mn.graph
        merge_graphs(g)
        nodes.extend(list(g.nodes()))
        edges.extend(list(g.edges()))
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    graph.remove_nodes_from(model_nodes)
