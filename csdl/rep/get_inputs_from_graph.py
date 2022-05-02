from networkx import DiGraph
from csdl.lang.input import Input
from csdl.rep.variable_node import VariableNode


def get_inputs_from_graph(graph: DiGraph) -> list[VariableNode]:
    return list(
        filter(
            lambda x: isinstance(x, VariableNode) and isinstance(
                x.var, Input), graph.nodes()), )
