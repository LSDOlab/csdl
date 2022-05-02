from networkx import DiGraph
from csdl.lang.implicit_operation import ImplicitOperation
from csdl.rep.operation_node import OperationNode


def get_implicit_operations_from_graph(
        graph: DiGraph) -> list[OperationNode]:
    return list(
        filter(
            lambda x: isinstance(x.op, ImplicitOperation),
            filter(
                lambda x: isinstance(x, OperationNode) and graph.
                out_degree(x) == 0,
                graph.nodes(),
            ),
        ))
