from networkx import DiGraph
from csdl.rep.model_node import ModelNode


def get_model_nodes_from_graph(graph: DiGraph) -> list[ModelNode]:
    return list(
        filter(lambda x: isinstance(x, ModelNode), graph.nodes()))
