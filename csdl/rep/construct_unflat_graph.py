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
from typing import List, Dict, Set, Tuple
from copy import copy


def in_cycles(cycles: list[Set[str]], name: str):
    for c in cycles:
        if name in c:
            print(
                "{} was already found in a cycle; skipping to next branch"
                .format(name))
            return True
    return False


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
    # TODO: break cycles
    if isinstance(node, Variable):
        if node.name not in nodes.keys():
            nodes[node.name] = VariableNode(node)
    elif isinstance(node, Operation):
        if node.name not in nodes.keys():
            nodes[node.name] = OperationNode(node)
    for predecessor in node.dependencies:
        _construct_graph_this_level(graph, nodes, predecessor)
        try:
            a = nodes[predecessor.name]
        except:
            print('nodes', nodes.keys())
            print('PROBLEM', predecessor.name)
            exit()
        b = nodes[node.name]
        graph.add_edge(a, b)


def build_dag(
    dag: DiGraph,
    nodes: Dict[str, Node],
    edges: Dict[str, list[str]],
    node: Node,
    path: list[str] = [],
    cycles: list[Set[str]] = [],
) -> Tuple[list[Set[str]], Set[str], str | None]:
    culprit: str | None = None
    current_cycle = set()
    for predecessor in node.dependencies:
        if predecessor.name in path:
            return cycles, set(predecessor.name), predecessor.name

        # check if this node is on a cycle and skip searching its
        # children if it is
        if in_cycles(cycles, predecessor.name):
            # TODO: what to do here?
            long_cycle = set()
            for cycle in cycles:
                if predecessor.name in cycle:
                    long_cycle.update(cycle)
            cycles.append(long_cycle)
            continue
        else:
            # continue search according to DFS strategy, keep track of
            # path traversed
            path.append(predecessor.name)
            dag.add_edge(predecessor, node)
            # explore branches from this node
            cycles, current_cycle, culprit = build_dag(
                dag,
                nodes,
                edges,
                predecessor,
                path=path,
                cycles=cycles,
            )

            if culprit is not None:
                # cycle detected downstream
                if culprit != predecessor.name:
                    # build cycle
                    current_cycle.add(predecessor.name)
                else:
                    # end of cycle
                    # no culprit to indicate a cycle
                    culprit = None
                    # update record of cycles
                    cycles.append(copy(current_cycle))
                    current_cycle = set()
            path.pop()
    return cycles, current_cycle, culprit


def construct_unflat_graph(
    inputs: List[Input],
    registered_outputs: List[Output],
    subgraphs: List[Subgraph],
    recursive: bool = True,
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
            mn = ModelNode(s.name, s.submodel, s.promotes)
            nodes[s.name] = mn
            graph.add_node(mn)
            if recursive is True:
                mn.graph = construct_unflat_graph(
                    s.submodel.inputs,
                    s.submodel.registered_outputs,
                    s.submodel.subgraphs,
                )

    # construct graph from the variables, operations, and subgraphs at
    # this level of the hierarchy
    for inp in inputs:
        if inp.name not in nodes.keys():
            nodes[inp.name] = VariableNode(inp)
        if nodes[inp.name] not in graph.nodes():
            graph.add_node(nodes[inp.name])
    for r in registered_outputs:
        _construct_graph_this_level(graph, nodes, r)


    return graph
