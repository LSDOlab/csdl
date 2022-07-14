from typing import Tuple, Set, Dict, List
from networkx import DiGraph, draw, contracted_nodes
try:
    from csdl.rep.construct_flat_graph import GraphWithMetadata
except ImportError:
    pass
from csdl.rep.variable_node import VariableNode
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.variable import Variable
from csdl.lang.custom_operation import CustomOperation
import numpy as np
import warnings


def find_unique_node(
    container: 'GraphWithMetadata',
    name: str,
    namespace: str,
) -> VariableNode:
    """
    Find unique name of a variable in a connection specified by user.
    """
    given_name = prepend_namespace(namespace, name)

    # if given name is promoted, return node
    if given_name in container.promoted_to_node:
        return container.promoted_to_node[given_name]

    # if given name is promoted, return node
    elif given_name in container.unpromoted_to_node:
        return container.unpromoted_to_node[given_name]

    # If we can't find the variable, return error
    raise KeyError(
        "{} is not a user defined variable name in the model".format(
            given_name))


def merge_connections(
    container: 'GraphWithMetadata',
    connections: List[Tuple[str, str, str]],
    promoted_to_unpromoted: Dict[str, Set[str]],
    unpromoted_to_promoted: Dict[str, str],
) -> 'GraphWithMetadata':
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
    promoted_to_node: Dict[str, VariableNode] = {}
    unpromoted_to_node: Dict[str, VariableNode] = {}
    for var_node in container.graph.nodes:

        # ignore if node is an operation
        if not isinstance(var_node, VariableNode):
            continue

        var_node.connected_to = set()
        num_pred = len(list(container.graph.predecessors(var_node)))

        if num_pred > 1:
            raise KeyError(f'variable {var_node.name} has multiple predecessors')

        # ignore if node name starts with an underscore
        if var_node.var.name == var_node.var._id:
            # if var_node.var.name[0] == '_':
            continue

        # Each promoted name maps to a unique variable
        promoted_name = prepend_namespace(var_node.namespace,
                                          var_node.name)

        # Error for devs, promoted names in graph should be in
        # promoted_to_unpromoted
        # Possible causes for this error
        # - Output of custom operation not registered (will be fixed in
        #   the future)
        # - User used a variable from one Model class in another Model
        #   instance (Is there a way to notify user clearly, earlier?)
        if promoted_name not in promoted_to_unpromoted.keys():
            # If variable is an output of a custom operation, it will
            # have the same name as the output within the custom
            # operation, even if the user does not register the output.
            # In this case, the variable name will not appear in either
            # promoted_to_unpromoted, or unpromoted_to_promoted.
            # However, if a different variable is registered with the
            # same name, we still have to check that it is in the
            # dictionary, so we check if the variable depends on a
            # custom operation only after we cannot find the name in
            # promoted_to_unpromoted.
            if not isinstance(
                    list(container.graph.predecessors(var_node))[0].op,
                    CustomOperation):
                raise KeyError(
                    f'\'{promoted_name}\' not found in promoted_to_unpromoted. This error is usually an indication that a `Model` was defined "inline" and one of its variables were used outside that `Model` instance.'
                )
            else:
                continue

        promoted_to_node[promoted_name] = var_node

        if promoted_name in promoted_to_unpromoted.keys():
            for unpromoted_name in promoted_to_unpromoted[
                    promoted_name]:
                # Error for devs, unpromoted_name in graph should not already be in unpromoted_to_node
                if unpromoted_name in unpromoted_to_node.keys():
                    raise KeyError(
                        f'\'{unpromoted_name}\' already in unpromoted_to_node'
                    )

                # Error for devs, unpromoted_name in graph should not already be in unpromoted_to_node
                if unpromoted_name not in unpromoted_to_promoted.keys():
                    raise KeyError(
                        f'\'{unpromoted_name}\' not found in unpromoted_to_promoted'
                    )

                # Set mapping
                unpromoted_to_node[unpromoted_name] = var_node

    container.promoted_to_node = promoted_to_node
    container.unpromoted_to_node = unpromoted_to_node
    container.connected_tgt_nodes_to_source_nodes = {}

    # draw(container.graph, with_labels = True)
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
        src_node = find_unique_node(container, src_name,
                                    model_namespace)

        # 2)
        tgt_node = find_unique_node(container, tgt_name,
                                    model_namespace)
        # Check to make sure target has not already been connected
        if tgt_node in container.connected_tgt_nodes_to_source_nodes:
            clashing_src_name = container.connected_tgt_nodes_to_source_nodes[
                tgt_node].var.name
            raise KeyError(
                f'connection target \'{tgt_name}\' has already been connected to {clashing_src_name}'
            )

        # 3)
        # check source type
        if not isinstance(src_node.var, (Input, Output)):
            raise ValueError(
                f'connection source \'{src_name}\' is not an input/output.'
            )
        # check target type
        if not isinstance(tgt_node.var, (DeclaredVariable, Variable)):
            print(tgt_node, tgt_node.var)
            raise ValueError(
                f'connection target \'{tgt_name}\' is not a declared variable.'
            )
        else:
            tgt_in_degree = container.graph.in_degree(tgt_node)
            # check to make sure target doesn't already depend on another variable.
            if tgt_in_degree != 0:
                tgt_predecessors = list(
                    container.graph.predecessors(tgt_node))
                raise ValueError(
                    f'connection target \'{tgt_name}\' already has dependencies {tgt_predecessors}.'
                )
        # check shapes
        if src_node.var.shape != tgt_node.var.shape:

            # backend can reshape if the sizes are the same. otherwise, raise an error.
            if np.prod(src_node.var.shape) != np.prod(tgt_node.var.shape):
                raise ValueError(f'connection source \'{src_name}\' size is not equal to connection target \'{tgt_name}\' size. {src_node.var.shape} != {tgt_node.var.shape}.')
            else:
                warnings.warn(f'connection source \'{src_name}\' shape is not equal to connection target \'{tgt_name}\' shape. {src_node.var.shape} != {tgt_node.var.shape}.')
            # raise ValueError(
            #     f'connection source \'{src_name}\' shape is not equal to connection target \'{tgt_name}\' shape. {src_node.var.shape} != {tgt_node.var.shape}'
            # )

        # 4) Merge nodes
        contracted_nodes(
            container.graph,
            src_node,
            tgt_node,
            self_loops=False,
            copy=False,
        )

        # 5) Update mapping of connected targets --> source
        container.connected_tgt_nodes_to_source_nodes[
            tgt_node] = src_node

        # For each connection source, add target variable node
        src_node.connected_to.add(tgt_node)

    return container
