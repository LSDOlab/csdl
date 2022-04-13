from networkx import DiGraph

from csdl.core.output import Output
from csdl.core.variable import Variable
from csdl.core.operation import Operation
from csdl.core.subgraph import Subgraph
from csdl.ir.node import Node
from csdl.ir.variable_node import VariableNode
from csdl.ir.operation_node import OperationNode
from csdl.ir.model_node import ModelNode
from typing import List, Dict


def construct_graph_this_level(
    graph: DiGraph,
    nodes: Dict[str, Node],
    node: Operation | Variable | Output | Subgraph,
):
    """
    Construct the graph for a single model in the model hierarchy where
    variables, operations, and submodels are represented by nodes and
    their dependency relationships represented by edges
    """
    if isinstance(node, Variable):
        if node.name not in nodes.keys():
            nodes[node.name] = VariableNode(node)
    elif isinstance(node, Operation):
        if node.name not in nodes.keys():
            nodes[node.name] = OperationNode(node)
    graph.add_edge(nodes[node.name], node)
    for predecessor in node.dependencies:
        construct_graph_this_level(graph, nodes, predecessor)


def construct_graph(
    registered_nodes: List[Output],
    subgraphs: List[Subgraph],
) -> DiGraph:
    """
    Construct the graph for the entire model hierarchy where variables,
    operations, and submodels are represented by nodes and their
    dependency relationships represented by edges
    """
    nodes: Dict[str, Node] = dict()
    graph = DiGraph()

    # recursively add models to intermediate representation from the
    # bottom of the hierarchy to the top
    for node in subgraphs:
        # some subgraphs will not be connected to variables in the parent
        # model prior to flattening the graph
        if node.name not in nodes.keys():
            mn = ModelNode(node.submodel)
            nodes[node.name] = mn
            graph.add_node(mn)

        mn = nodes[node.name]
        if isinstance(mn, ModelNode):
            mn.graph = construct_graph(
                node.submodel.registered_outputs,
                node.submodel.subgraphs,
            )

    # construct graph at this level of the hierarchy
    for node in registered_nodes:
        construct_graph_this_level(graph, nodes, node)
    return graph
