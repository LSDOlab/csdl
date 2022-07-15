from networkx import DiGraph

from csdl.lang.output import Output
from csdl.lang.input import Input
from csdl.lang.variable import Variable
from csdl.lang.operation import Operation
from csdl.lang.implicit_operation import ImplicitOperation
from csdl.lang.subgraph import Subgraph
from csdl.lang.node import Node
from csdl.rep.ir_node import IRNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.implicit_operation_node import ImplicitOperationNode
from csdl.rep.model_node import ModelNode
from csdl.rep.get_nodes import get_model_nodes, get_src_nodes, get_tgt_nodes, get_var_nodes, get_operation_nodes, get_implicit_operation_nodes
from csdl.utils.prepend_namespace import prepend_namespace
from typing import List, Dict, Set, Union, List
from warnings import warn


def _construct_graph_this_level(
    graph: DiGraph,
    nodes: Dict[str, Union[VariableNode, OperationNode, ModelNode]],
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
            if isinstance(node, ImplicitOperation):
                nodes[node.name] = ImplicitOperationNode(node)
            else:
                nodes[node.name] = OperationNode(node)
    for predecessor in node.dependencies:

        # _construct_graph_this_level(graph, nodes, predecessor)
        # # adding redundant edge will not affect graph structure
        # graph.add_edge(nodes[predecessor.name], nodes[node.name])

        if predecessor.name not in nodes:
            _construct_graph_this_level(graph, nodes, predecessor)
            # adding redundant edge will not affect graph structure
            graph.add_edge(nodes[predecessor.name], nodes[node.name])
        elif node.name not in nodes:
            _construct_graph_this_level(graph, nodes, predecessor)
            # adding redundant edge will not affect graph structure
            graph.add_edge(nodes[predecessor.name], nodes[node.name])
        elif not graph.has_edge(nodes[predecessor.name], nodes[node.name]):

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
    nodes: Dict[str, Union[VariableNode, OperationNode, ModelNode]] = dict()
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
    input_nodes = {inp.name: VariableNode(inp) for inp in inputs}
    nodes.update(input_nodes)
    graph.add_nodes_from(input_nodes.values())
    # for inp in inputs:
    #     if inp.name not in nodes.keys():
    #         nodes[inp.name] = VariableNode(inp)
    #     if nodes[inp.name] not in graph.nodes():
    #         graph.add_node(nodes[inp.name])

    # add nodes that outputs depend on for this model
    for r in registered_outputs:
        _construct_graph_this_level(graph, nodes, r)

    return graph


def find_cycles_among_models(
    graph: DiGraph,
    nodes: List[IRNode],
    path: List[IRNode] = [],
    cycles: List[Set[IRNode]] = [],
) -> List[Set[IRNode]]:
    for node in nodes:
        path_as_set = set()
        # if name in path, store new cycle; this is not the most general
        # way to detect new cycles, but if a new cycle can be detected
        # this way, it is less expensive than the most general way
        if node in path:
            # TODO: remove edge from predecessor to node in graph
            path_as_set = set(path[path.index(node):])
            cycles.append(path_as_set)
            return cycles

        # continue search according to DFS strategy, keep track of
        # path traversed
        path.append(node)
        cycles = find_cycles_among_models(
            graph,
            graph.successors(node),
            path=path,
            cycles=cycles,
        )
        path.pop()
    return cycles


def construct_unflat_graph(graph: DiGraph, namespace: str = '') -> DiGraph:
    """
    Construct the intermediate representation as a graph with nodes
    represented by `VariableNode`, `OperationNode`, and `ModelNode`
    objects.
    `ModelNode` objects contain a graph for a submodel.
    The intermediate representation graph and all graphs contained in a
    `ModelNode` are implemented as a networkx `DiGraph`.
    """
    model_nodes: List[ModelNode] = get_model_nodes(graph)
    for mn in model_nodes:
        _ = construct_unflat_graph(mn.graph, namespace=prepend_namespace(namespace, mn.namespace))
    var_nodes = get_var_nodes(graph)

    # src -> model
    src_nodes = get_src_nodes(var_nodes)
    for src in src_nodes:
        src_path = prepend_namespace(src.namespace, src.name)
        for mn in model_nodes:
            child_var_nodes = get_var_nodes(mn.graph)
            child_tgt_nodes = get_tgt_nodes(child_var_nodes)
            for tgt in child_tgt_nodes:
                tgt_path = prepend_namespace(tgt.namespace, tgt.name)
                if tgt_path == src_path:
                    graph.add_edge(src, mn)
                    break

    # tgt -> model
    tgt_nodes = get_tgt_nodes(var_nodes)
    for tgt in tgt_nodes:
        tgt_path = prepend_namespace(tgt.namespace, tgt.name)
        for mn in model_nodes:
            child_var_nodes = get_var_nodes(mn.graph)
            child_src_nodes = get_src_nodes(child_var_nodes)
            for src in child_src_nodes:
                src_path = prepend_namespace(src.namespace, src.name)
                if tgt_path == src_path:
                    graph.add_edge(mn, tgt)
                    break

    # model -> model
    for mn1 in model_nodes:
        for mn2 in model_nodes:
            if mn1 is not mn2:
                var_nodes1 = get_var_nodes(mn1.graph)
                var_nodes2 = get_var_nodes(mn2.graph)
                src_nodes = get_src_nodes(var_nodes1)
                tgt_nodes = get_tgt_nodes(var_nodes2)
                for src in src_nodes:
                    src_path = prepend_namespace(src.namespace,
                                                 src.name)
                    for tgt in tgt_nodes:
                        tgt_path = prepend_namespace(
                            tgt.namespace, tgt.name)
                        if tgt_path == src_path:
                            graph.add_edge(mn1, mn2)

    # TODO: ensure that redundant cycles are not recorded
    cycles: List[Set[IRNode]] = []
    for mn in model_nodes:
        cycles.extend(find_cycles_among_models(graph, [mn]))
    if len(cycles) > 1:
        warn("Model {} forms at least one cycle between two or more submodels. Cycles present in the unflat graph will affect performance if using a CSDL compiler back end that uses the unflat graph representation. Cycles present are, {}.\nIf using a CSDL compiler back end that uses the flattened graph representation, disregard this warning.".format(namespace, cycles))

    return graph
