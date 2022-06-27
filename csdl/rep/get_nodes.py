from csdl.rep.variable_node import VariableNode
from csdl.rep.model_node import ModelNode
from csdl.lang.variable import Variable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable
from networkx import DiGraph


def get_var_nodes(graph: DiGraph) -> list[VariableNode]:
    return list(
        filter(lambda x: isinstance(x, VariableNode), graph.nodes()))


def get_tgt_nodes(var_nodes: list[VariableNode]) -> list[VariableNode]:
    return list(
        filter(lambda x: isinstance(x.var, DeclaredVariable),
               var_nodes))


def get_src_nodes(var_nodes: list[VariableNode]) -> list[VariableNode]:
    return list(
        filter(lambda x: isinstance(x.var, (Input, Output)), var_nodes))


def get_model_nodes(graph: DiGraph) -> list[ModelNode]:
    return list(
        filter(lambda x: isinstance(x, ModelNode), graph.nodes()))


def get_input_nodes(graph: DiGraph) -> list[VariableNode]:
    return list(
        filter(lambda x: isinstance(x.var, Input),
               get_var_nodes(graph)))
