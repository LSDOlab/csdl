from typing import Tuple, Set
from networkx import DiGraph, draw, contracted_nodes
from csdl.rep.variable_node import VariableNode
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.variable import Variable


def find_unique_node(
    graph: DiGraph,
    name: str,
    namespace: str,
) -> VariableNode:
    """
    Find unique name of a variable in a connection specified by user.
    """
    given_name = prepend_namespace(namespace, name)

    # if given name is promoted, return node
    if given_name in graph.promoted_to_node:
        return graph.promoted_to_node[given_name]

    # if given name is promoted, return node
    elif given_name in graph.unpromoted_to_node:
        return graph.unpromoted_to_node[given_name]

    # If we can't find the variable, return error
    raise KeyError(
        "{} is not a user defined variable name in the model".format(
            given_name))


def merge_connections(
    flat_graph: DiGraph,
    connections: list[Tuple[str, str, str]],
    promoted_to_unpromoted,
    unpromoted_to_promoted,
):
    '''
    appropriate nodes in connections are merged in flat_graph.

    Parameters:
    flat_graph: a networkx DiGraph containing only variable nodes and operation nodes.
    connections: list containing source, target and model of connection.
    '''

    # print(promoted_to_unpromoted)
    # for p in promoted_to_unpromoted:
    #     print(p, promoted_to_unpromoted[p])
    # print(unpromoted_to_promoted)

    # Create mapping of
    # -- promoted names to nodes in the flat graph
    # -- unpromoted names to nodes in the flat graph
    promoted_to_node = {}
    unpromoted_to_node = {}
    for var_node in flat_graph.nodes:

        # ignore if node is an operation
        if not isinstance(var_node, VariableNode):
            continue

        # ignore if node name starts with an underscore
        if var_node.var.name[0] == '_':
            continue

        # Each promoted name maps to a unique variable
        promoted_name = prepend_namespace(var_node.namespace, var_node.name)
        promoted_to_node[promoted_name] = var_node

        # Error for devs, promoted names in graph should be in promoted_to_unpromoted
        if promoted_name not in promoted_to_unpromoted:
            raise KeyError(f'\'{promoted_name}\' not found in promoted_to_unpromoted')

        for unpromoted_name in promoted_to_unpromoted[promoted_name]:
            # Error for devs, unpromoted_name in graph should not already be in unpromoted_to_node
            if unpromoted_name in unpromoted_to_node:
                raise KeyError(f'\'{unpromoted_name}\' already in unpromoted_to_node')

            # Error for devs, unpromoted_name in graph should not already be in unpromoted_to_node
            if unpromoted_name not in unpromoted_to_promoted:
                raise KeyError(f'\'{unpromoted_name}\' not found in unpromoted_to_promoted')

            # Set mapping
            unpromoted_to_node[unpromoted_name] = var_node

    flat_graph.promoted_to_node = promoted_to_node
    flat_graph.unpromoted_to_node = unpromoted_to_node
    flat_graph.connected_tgt_nodes_to_source_nodes = {}

    # draw(flat_graph, with_labels = True)
    # import matplotlib.pyplot as plt
    # plt.show()

    # Loop through each connection.
    for src_name, tgt_name, model_namespace in connections:
        # 1) Locate source in graph
        # --- Check to make sure source exists
        # 2) Locate target in graph
        # --- Check to make sure target exists
        # --- Check to make sure target has not already been connected
        # 3) Check to make sure connection is allowed
        # --- Check to make sure source is input/output
        # --- Check to make sure target is declared variable
        # --- Check to make sure target has no in-edges
        # --- Check to make sure shapes match
        # 4) Merge nodes
        # 5) Update mapping of connected targets --> source

        # 1)
        src_node = find_unique_node(flat_graph, src_name, model_namespace)

        # 2)
        tgt_node = find_unique_node(flat_graph, tgt_name, model_namespace)
        # Check to make sure target has not already been connected
        if tgt_node in flat_graph.connected_tgt_nodes_to_source_nodes:
            clashing_src_name = flat_graph.connected_tgt_nodes_to_source_nodes[tgt_node].var.name
            raise KeyError(f'connection target \'{tgt_name}\' has already been connected to {clashing_src_name}')

        # 3)
        # check source type
        if not isinstance(src_node.var, (Input, Output)):
            raise ValueError(f'connection source \'{src_name}\' is not an input/output.')
        # check target type
        if not isinstance(tgt_node.var, (DeclaredVariable, Variable)):
            print(tgt_node, tgt_node.var)
            raise ValueError(f'connection target \'{tgt_name}\' is not a declared variable.')
        else:
            tgt_in_degree = flat_graph.in_degree(tgt_node)
            # check to make sure target doesn't already depend on another variable.
            if tgt_in_degree != 0:
                tgt_predecessors = list(flat_graph.predecessors(tgt_node))
                raise ValueError(f'connection target \'{tgt_name}\' already has dependencies {tgt_predecessors}.')
        # check shapes
        if src_node.var.shape != tgt_node.var.shape:
            raise ValueError(f'connection source \'{src_name}\' shape is not equal to connection target \'{tgt_name}\' shape. {src_node.var.shape} != {tgt_node.var.shape}')

        # 4) Merge nodes
        contracted_nodes(
            flat_graph,
            src_node,
            tgt_node,
            self_loops=False,
            copy=False,
        )

        # 5) Update mapping of connected targets --> source
        flat_graph.connected_tgt_nodes_to_source_nodes[tgt_node] = src_node
