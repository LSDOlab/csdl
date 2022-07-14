from networkx import DiGraph
from csdl.lang.output import Output
from csdl.rep.variable_node import VariableNode
from typing import List

def get_registered_outputs_from_graph(
        graph: DiGraph) -> List[VariableNode]:
    return list(
        filter(
            lambda x: isinstance(x.var, Output) and x.var.name != x.var.
            _id,
            filter(
                lambda x: isinstance(x, VariableNode) and graph.
                out_degree(x) == 0,
                graph.nodes(),
            ),
        ))
