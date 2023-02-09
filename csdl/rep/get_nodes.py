from csdl.rep.variable_node import VariableNode
from csdl.rep.model_node import ModelNode
from csdl.rep.operation_node import OperationNode
try:
    from csdl.rep.implicit_operation_node import ImplicitOperationNode
except ImportError:
    pass
from csdl.lang.variable import Variable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable
from networkx import DiGraph
from typing import List

# def _collect_variable_nodes(self):
#     # TODO: need to collect these in order?
#     variables = []
#     implicit_operation_nodes = list(
#         filter(lambda x: isinstance(x, ImplicitOperationNode),
#                self._operation_nodes))
#     for op in implicit_operation_nodes:
#         variables.extend(op.rep._collect_variable_nodes())
#     return copy(self._variable_nodes) + variables


def get_var_nodes(
    graph: DiGraph,
    implicit: bool = False,
) -> List[VariableNode]:
    if implicit is False:
        return list(
            filter(lambda x: isinstance(x, VariableNode),
                   graph.nodes()))
    # TODO: need to collect these in order?
    variable_nodes = list(
        filter(lambda x: isinstance(x, VariableNode), graph.nodes()))
    implicit_operation_nodes: List[ImplicitOperationNode] = list(
        filter(lambda x: isinstance(x, ImplicitOperationNode),
               graph.nodes()))
    for ii in implicit_operation_nodes:
        variable_nodes.extend(
            get_var_nodes(ii.rep.flat_graph, implicit=True))
    return variable_nodes


def get_tgt_nodes(var_nodes: List[VariableNode]) -> List[VariableNode]:
    return list(
        filter(lambda x: isinstance(x.var, DeclaredVariable),
               var_nodes))


def get_src_nodes(var_nodes: List[VariableNode]) -> List[VariableNode]:
    return list(
        filter(lambda x: isinstance(x.var, (Input, Output)), var_nodes))


def get_model_nodes(graph: DiGraph) -> List[ModelNode]:
    return list(
        filter(lambda x: isinstance(x, ModelNode), graph.nodes()))


def get_input_nodes(
        var_nodes: List[VariableNode]) -> List[VariableNode]:
    return list(filter(lambda x: isinstance(x.var, Input), var_nodes))


def get_output_nodes(
        var_nodes: List[VariableNode]) -> List[VariableNode]:
    return list(filter(lambda x: isinstance(x.var, Output), var_nodes))


def get_operation_nodes(
    graph: DiGraph,
    implicit: bool = False,
) -> List[OperationNode]:
    if implicit is False:
        return list(
            filter(lambda x: isinstance(x, OperationNode),
                   graph.nodes()))
    # TODO: need to collect these in order?
    operation_nodes = list(
        filter(lambda x: isinstance(x, OperationNode), graph.nodes()))
    implicit_operation_nodes: List[ImplicitOperationNode] = list(
        filter(lambda x: isinstance(x, ImplicitOperationNode),
               graph.nodes()))
    for ii in implicit_operation_nodes:
        operation_nodes.extend(
            get_operation_nodes(ii.rep.flat_graph, implicit=True))
    return operation_nodes


def get_implicit_operation_nodes(
        graph: DiGraph) -> List['ImplicitOperationNode']:
    return list(
        filter(lambda x: isinstance(x, ImplicitOperationNode),
               graph.nodes()))
