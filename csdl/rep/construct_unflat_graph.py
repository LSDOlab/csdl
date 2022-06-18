from lib2to3.pgen2.driver import load_grammar
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
try:
    from csdl.lang.model import Model
except ImportError:
    pass


def add_dependencies_for_models(model: 'Model'):
    # TODO: check dependencies

    # add dependency relationship representing data flowing from local
    # source to child model
    for s in model.subgraphs:
        for local_name, unpromoted_names in model.promoted_names_to_unpromoted_names.items(
        ):
            io: list[Input | Output] = []
            io.extend(model.inputs)
            io.extend(model.registered_outputs)
            src = list(filter(lambda x: x.name == local_name, io))
            if len(src) > 0:
                for unpromoted_name in unpromoted_names:
                    # TODO: rsplit unpromoted_name
                    if unpromoted_name in s.submodel.promoted_names_to_unpromoted_names.keys(
                    ):
                        s.add_dependency_node(src[0])
                        src[0].add_dependent_node(s)

    # add dependency relationship representing data flowing from child
    # model to local target
    for s in model.subgraphs:
        for local_name, unpromoted_names in model.promoted_names_to_unpromoted_names.items(
        ):
            tgt = list(
                filter(lambda x: x.name == local_name,
                       model.registered_outputs))
            if len(tgt) > 0:
                for unpromoted_name in unpromoted_names:
                    a = unpromoted_name.rsplit('.')
                    if len(a) > 1:
                        if a[1:] in s.submodel.promoted_names_to_unpromoted_names.keys(
                        ):
                            tgt[0].add_dependency_node(s)
                            s.add_dependent_node(tgt[0])

    # add dependency relationship representing data flowing from local
    # target to target with same name and shape in child model that has
    # been promoted to this model
    for s in model.subgraphs:
        for local_name, tgt in locally_defined_targets.items():
            for lower_level_name in promoted_targets_from_children.keys(
            ):
                if local_name == lower_level_name:
                    s.add_dependency_node(tgt)
                    tgt.add_dependent_node(s)

    # add dependency relationship representing data flowing between
    # child models
    for a in model.subgraphs:
        for b in model.subgraphs:
            if a is not b:
                for name in a.submodel.promoted_source_shapes.keys():
                    if name in b.submodel.promoted_target_shapes.keys():
                        b.add_dependency_node(a)
                        a.add_dependent_node(b)

    # TODO: make sure users can't form cycles by defining terms in
    # between concatenation assignments; e.g.
    # a[1] = b
    # c = 2*a[1]
    # a[2] = c
    # then insert condition so that this only runs for models containing
    # models
    check_for_cycles(model, namespace, 'promotions')


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
    if isinstance(node, Variable):
        if node.name not in nodes.keys():
            nodes[node.name] = VariableNode(node)
    elif isinstance(node, Operation):
        if node.name not in nodes.keys():
            nodes[node.name] = OperationNode(node)
    for predecessor in node.dependencies:
        _construct_graph_this_level(graph, nodes, predecessor)
        # adding redundant edge will not affect graph structure
        graph.add_edge(nodes[predecessor.name], nodes[node.name])


def construct_graphs_all_models(
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

    # add models to graph for this model
    for s in subgraphs:
        if s.name not in nodes.keys():
            mn = ModelNode(s.name, s.submodel, s.promotes)
            nodes[s.name] = mn
            graph.add_node(mn)
            mn.graph = construct_graphs_all_models(
                s.submodel.inputs,
                s.submodel.registered_outputs,
                s.submodel.subgraphs,
            )

    # add variables and operations to the graph for this model

    # add inputs to the graph for this model
    for inp in inputs:
        if inp.name not in nodes.keys():
            nodes[inp.name] = VariableNode(inp)
        if nodes[inp.name] not in graph.nodes():
            graph.add_node(nodes[inp.name])

    # add nodes that outputs depend on for this model
    for r in registered_outputs:
        _construct_graph_this_level(graph, nodes, r)

    return graph


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
