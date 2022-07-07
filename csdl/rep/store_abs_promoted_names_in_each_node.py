from typing import Dict, List
from networkx import DiGraph
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode


def store_abs_promoted_names_in_each_node(
    graph: DiGraph,
    unpromoted_to_promoted: Dict[str, str],
):
    vars: List[VariableNode] = list(
        filter(lambda x: isinstance(x, VariableNode), graph.nodes()))
    ops: List[OperationNode] = list(
        filter(lambda x: isinstance(x, OperationNode), graph.nodes()))
    for v in vars:
        if v.var.abs_name is not None:
            v.var.abs_prom_name = unpromoted_to_promoted[v.var.abs_name]
    # for op in ops:
    #     if op.op.abs_name is not None:
    #         op.op.abs_prom_name = unpromoted_to_promoted[op.op.abs_name]
