from networkx import DiGraph

from csdl.lang.output import Output
from csdl.lang.input import Input
from csdl.lang.variable import Variable
from csdl.lang.operation import Operation
from csdl.lang.subgraph import Subgraph
from csdl.lang.node import Node
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.model_node import ModelNode
from typing import List, Dict


def _construct_graph_this_level(
    graph: DiGraph,
    nodes: Dict[str, VariableNode | OperationNode | ModelNode],
    node: Node,
):
    """
    Construct the graph to store in a `ModelNode` object.
    This function is applied recursively from the registered outputs in
    the model to the inputs and declared variables in the model in order
    to build the graph.
    """
    if isinstance(node, Variable):
        if node.name not in nodes.keys():
            nodes[node.name] = VariableNode(node)
    elif isinstance(node, Operation):
        if node.name not in nodes.keys():
            nodes[node.name] = OperationNode(node)
    for predecessor in node.dependencies:
        _construct_graph_this_level(graph, nodes, predecessor)
        a = nodes[predecessor.name]
        b = nodes[node.name]
        graph.add_edge(a, b)


def construct_unflat_graph(
    inputs: List[Input],
    registered_outputs: List[Output],
    subgraphs: List[Subgraph],
) -> DiGraph:
    """
    Construct the intermediate representation as a graph with nodes
    represented by `VariableNode`, `OperationNode`, and `ModelNode`
    objects.
    `ModelNode` objects contain a graph for a submodel.
    The intermediate representation graph and all graphs contained in a
    `ModelNode` are implemented as a networkx `DiGraph`.
    """
    nodes: Dict[str, VariableNode | OperationNode | ModelNode] = dict()
    graph = DiGraph()

    # recursively add models to intermediate representation from the
    # bottom of the hierarchy to the top
    for s in subgraphs:
        # some subgraphs will not be connected to variables in the
        # parent model prior to flattening the graph
        if s.name not in nodes.keys():
            mn = ModelNode(s.name, s.submodel)
            nodes[s.name] = mn
            graph.add_node(mn)

        # construct graph for each submodel and store in ModelNode
        n = nodes[s.name]
        if isinstance(n, ModelNode):
            n.graph = construct_unflat_graph(
                s.submodel.inputs,
                s.submodel.registered_outputs,
                s.submodel.subgraphs,
            )

    # construct graph from the variables, operations, ans subgraphs at
    # this level of the hierarchy
    for inp in inputs:
        if inp.name not in nodes.keys():
            nodes[inp.name] = VariableNode(inp)
        if nodes[inp.name] not in graph.nodes():
            graph.add_node(nodes[inp.name])
    for r in registered_outputs:
        _construct_graph_this_level(graph, nodes, r)
    return graph
