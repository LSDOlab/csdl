from networkx import DiGraph
from csdl.rep.model_node import ModelNode
from typing import List

def get_model_nodes_from_graph(graph: DiGraph) -> List[ModelNode]:
    return list(
        filter(lambda x: isinstance(x, ModelNode), graph.nodes()))
