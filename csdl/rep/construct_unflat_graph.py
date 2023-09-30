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
from typing import List, Dict, Set, Union, List, Tuple
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
        elif nodes[node.name].var is not node:
            # node.rep_node = nodes[node.name]
            node.rep_node.add_IR_mapping(nodes[node.name])

    elif isinstance(node, Operation):
        if node.name not in nodes.keys():
            if isinstance(node, ImplicitOperation):
                nodes[node.name] = ImplicitOperationNode(node)
            else:
                nodes[node.name] = OperationNode(node)
    for predecessor in node.dependencies:

        # Add edges between predecessor and node.
        # Function is called iteratively on predecessors of a node
        # and edges are added 'bottom up' from each registered output.
        """
        (x = reg.output)
          x       o       o       o
                  .
          x .......

          x       o ..... o       o
                  .
          x .......

          x       o ..... o ..... o
                  .
          x .......

          x       o ..... o <---- o
                  .
          x .......

          x       o <---- o <---- o
                  .
          x .......

          x       o <---- o <---- o
                  |
          x <------

          x ..... o <---- o <---- o
                  |
          x <------

          x <---- o <---- o <---- o
                  |
          x <------

          """

        # If an edge already exists between nodes, there
        # is no need to traverse down it again as all
        # downstream nodes have already been processed like the
        # last figure.

        # However, you get a weird edge case with multi-output
        # operations that aren't registered outputs:
        """
        (x = reg.output)
          o       o       o       o
                  .
          x .......

          o       o ..... o       o
                  .
          x .......

          o       o ..... o ..... o
                  .
          x .......

          o       o ..... o <---- o
                  .
          x .......

          o       o <---- o <---- o
                  .
          x .......

          o       o <---- o <---- o
                  |
          x <------

          """

        # The last edge isn't added because the top left node isn't
        # a registered output. This occurs when an operation outputs
        # two variables but the user only uses one in their calculation.
        # We can deal with this by manually
        # adding edges for multi-output operations.

        if predecessor.name not in nodes:
            _construct_graph_this_level(graph, nodes, predecessor)
            # adding redundant edge will not affect graph structure
            add_edge_to_graph(graph, nodes, predecessor.name, node.name)
        elif node.name not in nodes:
            _construct_graph_this_level(graph, nodes, predecessor)
            # adding redundant edge will not affect graph structure
            add_edge_to_graph(graph, nodes, predecessor.name, node.name)
        elif not graph.has_edge(nodes[predecessor.name],
                                nodes[node.name]):

            _construct_graph_this_level(graph, nodes, predecessor)
            # adding redundant edge will not affect graph structure
            add_edge_to_graph(graph, nodes, predecessor.name, node.name)


def add_edge_to_graph(
    graph: DiGraph,
    nodes: Dict[str, Union[VariableNode, OperationNode, ModelNode]],
    predecessor_name: str,
    node_name: str,
):
    """
    given a node and it's predecessor, add the edge to graph.
    """
    predecessor_instance = nodes[predecessor_name]
    graph.add_edge(predecessor_instance, nodes[node_name])

    # edge case if node is a multi-output operation. as discussed in
    # def _construct_graph_this_level:
    """
        manually add this edge
           /
          /
    o <==== o <---- o <---- o
            |
    x <------

    """
    if isinstance(predecessor_instance, OperationNode):
        # NEW: honestly not sure if this is correct
        # If this operation does not have the attribute out_left:
        if not hasattr(predecessor_instance, 'out_left'):
            predecessor_instance.out_left = list(predecessor_instance.op.outs)
            # print(predecessor_instance.out_left)
        # print(len(predecessor_instance.out_left), type(list(predecessor_instance.out_left)[0]))
        if nodes[node_name].var in set(predecessor_instance.out_left):
            predecessor_instance.out_left.remove(nodes[node_name].var)
            # predecessor_instance.out_left.pop()
        else:
            return

        if len(predecessor_instance.out_left) > 1:
            for successor in (predecessor_instance.out_left):
                _construct_graph_this_level(
                    graph,
                    nodes,
                    successor,
                )

        # OLD:
        # # add edges for multi-output operations
        # if len(predecessor_instance.op.outs) > 1:
        #     for successor in predecessor_instance.op.outs:
        #         _construct_graph_this_level(
        #             graph,
        #             nodes,
        #             successor,
        #         )


def construct_graphs_all_models(
    inputs: List[Input],
    registered_outputs: List[Output],
    declared_variables: List[Variable],
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
    nodes: Dict[str, Union[VariableNode, OperationNode,
                           ModelNode]] = dict()
    graph = DiGraph()

    graph.model_nodes = set()

    # add models to graph for this model
    for s in subgraphs:
        if s.name not in nodes.keys():
            mn = ModelNode(s.name, s.submodel, s.promotes)
            nodes[s.name] = mn
            graph.add_node(mn)
            mn.graph = construct_graphs_all_models(
                s.submodel.inputs,
                s.submodel.registered_outputs,
                s.submodel.declared_variables,
                s.submodel.subgraphs,
            )
            graph.model_nodes.add(mn)

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

    # OLD:
    # add nodes that outputs depend on for this model
    # for r in registered_outputs:
    #     _construct_graph_this_level(graph, nodes, r)

    # NEW: Unused declared variables should now be added to the graph
    construct_graph_this_model(graph, nodes, registered_outputs + declared_variables)

    return graph


def construct_graph_this_model(graph, nodes, leaves):
    # Takes an initialized graph of the

    # Create search queue
    bfs_queue = [leaf_node for leaf_node in leaves]
    queued_nodes = set(bfs_queue)

    # iteratively add variables
    while bfs_queue:

        # Node to add to graph
        # This node should only be in the queue once.
        csdl_node = bfs_queue.pop()
        graph_node = get_graph_node(nodes, csdl_node)

        # Add an edge for all predecessors of current node
        for predecessor_node in csdl_node.dependencies:

            # Add edge
            predecessor_graph_node = get_graph_node(nodes, predecessor_node)
            graph.add_edge(predecessor_graph_node, graph_node)

            # Process next node
            if predecessor_node in queued_nodes:
                continue
            bfs_queue.append(predecessor_node)
            queued_nodes.add(predecessor_node)

        # Add an edge for all successors of operation
        if isinstance(csdl_node, Operation):
            for successor_node in csdl_node.outs:

                # Add edge
                successor_graph_node = get_graph_node(nodes, successor_node)
                graph.add_edge(graph_node, successor_graph_node)

                # Process next node
                if successor_node in queued_nodes:
                    continue
                bfs_queue.append(successor_node)
                queued_nodes.add(successor_node)

    return


def get_graph_node(nodes, csdl_node):
    # Create new operation/variable node
    if isinstance(csdl_node, Variable):
        if csdl_node.name not in nodes.keys():
            nodes[csdl_node.name] = VariableNode(csdl_node)
        elif nodes[csdl_node.name].var is not csdl_node:
            # csdl_node.rep_node = nodes[csdl_node.name]
            csdl_node.add_IR_mapping(nodes[csdl_node.name])
    elif isinstance(csdl_node, Operation):
        if csdl_node.name not in nodes.keys():
            if isinstance(csdl_node, ImplicitOperation):
                nodes[csdl_node.name] = ImplicitOperationNode(csdl_node)
            else:
                nodes[csdl_node.name] = OperationNode(csdl_node)

    return nodes[csdl_node.name]


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


def model_has_target_for_source(mn: ModelNode, src_path: str) -> bool:
    child_var_nodes = get_var_nodes(mn.graph)
    child_tgt_nodes = get_tgt_nodes(child_var_nodes)
    for tgt in child_tgt_nodes:
        tgt_path = prepend_namespace(tgt.namespace, tgt.name)
        if tgt_path == src_path:
            return True
    model_nodes: List[ModelNode] = get_model_nodes(mn.graph)
    for child in model_nodes:
        if model_has_target_for_source(child, src_path):
            return True
    return False


def model_has_source_for_target(mn: ModelNode, tgt_path: str) -> bool:
    child_var_nodes = get_var_nodes(mn.graph)
    child_src_nodes = get_src_nodes(child_var_nodes)
    for src in child_src_nodes:
        src_path = prepend_namespace(src.namespace, src.name)
        if src_path == tgt_path:
            return True
    model_nodes: List[ModelNode] = get_model_nodes(mn.graph)
    for child in model_nodes:
        if model_has_source_for_target(child, tgt_path):
            return True
    return False


def add_source_model_edges(
    model_nodes: List[ModelNode],
    graph: DiGraph,
    src: VariableNode,
    src_path: str,
):
    for mn in model_nodes:
        if model_has_target_for_source(mn, src_path):
            graph.add_edge(src, mn)


def add_model_target_edges(
    model_nodes: List[ModelNode],
    graph: DiGraph,
    tgt: VariableNode,
    tgt_path: str,
):
    for mn in model_nodes:
        if model_has_source_for_target(mn, tgt_path):
            graph.add_edge(mn, tgt)


def construct_unflat_graph(
    graph: DiGraph,
    namespace: str = '',
) -> DiGraph:
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
        _ = construct_unflat_graph(
            mn.graph,
            namespace=prepend_namespace(namespace, mn.namespace),
        )
    var_nodes = get_var_nodes(graph)

    # src -> model
    src_nodes = get_src_nodes(var_nodes)
    for src in src_nodes:
        src_path = prepend_namespace(src.namespace, src.name)
        add_source_model_edges(model_nodes, graph, src, src_path)

    # model -> tgt
    tgt_nodes = get_src_nodes(var_nodes)
    for tgt in tgt_nodes:
        tgt_path = prepend_namespace(tgt.namespace, tgt.name)
        add_model_target_edges(model_nodes, graph, tgt, tgt_path)

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
                    if model_has_target_for_source(mn2, src_path):
                        graph.add_edge(mn1, mn2)
                for tgt in tgt_nodes:
                    tgt_path = prepend_namespace(tgt.namespace,
                                                 tgt.name)
                    if model_has_source_for_target(mn1, tgt_path):
                        graph.add_edge(mn1, mn2)

    # TODO: ensure that redundant cycles are not recorded
    cycles: List[Set[IRNode]] = []
    for mn in model_nodes:
        cycles.extend(find_cycles_among_models(graph, [mn]))
    if len(cycles) > 1:
        warn(
            "Model {} forms at least one cycle between two or more submodels. Cycles present in the unflat graph will affect performance if using a CSDL compiler back end that uses the unflat graph representation. Cycles present are, {}.\nIf using a CSDL compiler back end that uses the flattened graph representation, disregard this warning."
            .format(namespace, cycles))

    return graph
