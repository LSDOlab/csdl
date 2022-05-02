from typing import List
from csdl.rep.ir_node import IRNode
from csdl.rep.model_node import ModelNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.variable_node import VariableNode
from networkx import DiGraph


def assign_abs_name(node: IRNode, prefix: str | None = None):
    if prefix is not None:
        if isinstance(node, OperationNode):
            node.op.abs_name = prefix + '.' + node.op.name
        elif isinstance(node, VariableNode):
            node.var.abs_name = prefix + '.' + node.var.name


def store_abs_unpromoted_names_in_each_node(
    graph: DiGraph,
    prefix: str | None = None,
):
    """
    Store unpromoted name for each variable at each level in
    the model heirarchy.
    This is done temporarily so that when graphs are merged into one
    level of hierarchy, variable names do not clash.
    """
    model_nodes: List[ModelNode] = list(
        filter(lambda x: isinstance(x, ModelNode), graph.nodes()))
    # prepend namespace, starting from bottom of hierarchy <->
    for mn in model_nodes:
        store_abs_unpromoted_names_in_each_node(
            mn.graph,
            prefix=mn.name if prefix is None else prefix + '.' +
            mn.name,
        )
    for node in graph.nodes():
        assign_abs_name(node, prefix=prefix)
