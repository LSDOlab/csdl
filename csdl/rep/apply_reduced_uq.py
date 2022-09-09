from typing import Dict, Union, Tuple, Set, List
import numpy as np
from copy import copy
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from networkx import DiGraph
from csdl.operations.uq_tile import uq_tile
from csdl.lang.variable import Variable


def apply_reduced_uq(
    graph_meta: 'GraphMeta',
    rv_dict: Dict[Union[VariableNode], np.ndarray],
    dependence_data: Dict[Union[OperationNode, VariableNode], Dict[Union[OperationNode, VariableNode], bool]]
):

    # number of points for each random variable. (should be the same for all rvs)
    num_points = np.prod(rv_dict[list(rv_dict.keys())[0]].shape)

    # initialize new graph.
    old_graph = graph_meta.graph
    new_graph = old_graph.copy()

    # generate einsum_strings
    expansion_string_dict = build_einsums(rv_dict)

    # change csdl language dependencies
    for graph_node, node_dependence_dict in dependence_data.items():
        # if operation, check predecessors.
        if isinstance(graph_node, OperationNode):
            # if dependence of predecessors == dependence of operation, do nothing.
            # elif dependence of predecessors != dependence of operation, apply uq_expansion.

            # dictionary keys are nodes that need to be expanded with respect to dictionay values' random variable
            expansions_rv_dict = {}
            for predecessor in old_graph.predecessors(graph_node):
                expansions_rv_dict[predecessor] = []

                for rv_node, predecessor_node_depends_on_rv_node in dependence_data[predecessor].items():
                    graph_node_depends_on_rv_node = node_dependence_dict[rv_node]

                    # if the operation node depends on a rv that a predecessor does not depend on, need to expand the predecessor node.
                    if (not predecessor_node_depends_on_rv_node) and graph_node_depends_on_rv_node:
                        print(f'BUILD UQ EXPANSION ({rv_node.name}) between {predecessor.name} -->  {graph_node.name}')
                        expansions_rv_dict[predecessor].append(expansion_string_dict[rv_node])

            # Build the expanson nodes for each predecessor that needs expansion
            for predecessor in old_graph.predecessors(graph_node):
                expand_strings = expansions_rv_dict[predecessor]
                if expand_strings:
                    insert_expanson_node(
                        between=(predecessor, graph_node),  # nodes to insert between
                        graph=new_graph,  # graph to modify
                        expand_strings=expand_strings,  # expand in which direction?
                        num_points=num_points,  # number of points to expand to
                    )

            # if old_graph.predecessors[graph_node]:

        else:
            # if current node is a variable, expand based on dependence.
            # ex: if node depends on two variables, multiply first dimension size by num_points^2
            for rv_node, graph_node_depends_on_rv_node in node_dependence_dict.items():
                if graph_node_depends_on_rv_node:
                    new_shape = list(graph_node.var.shape)
                    new_shape[0] = new_shape[0]*num_points
                    graph_node.var.shape = tuple(new_shape)
                    if rv_node in rv_dict:
                        graph_node.var.val = rv_dict[rv_node]
                    else:
                        graph_node.var.val = np.zeros(graph_node.var.shape)
    # change csdl representation graph
    graph_meta.graph = new_graph
    return graph_meta


def build_einsums(rv_dict: Dict[VariableNode, np.ndarray]) -> Dict[VariableNode, str]:
    """
    returns the different expansion strings associated for each random variable.
    """
    expansion_string_dict = {}
    strings = [
        'i...,p...->pi...',
        'i...,p...->ip...',
    ]
    for i, rv in enumerate(rv_dict):
        expansion_string_dict[rv] = strings[i]

    return expansion_string_dict


def insert_expanson_node(
    between: Tuple[Union[VariableNode, OperationNode]],  # nodes to insert between
    graph: DiGraph,  # graph to modify
    expand_strings: List[str],  # expand in which direction?
    num_points: int,
):
    """
    Need to edit two representations:
    1) CSDL frontend node 'dependencies' and 'dependents'
    2) CSDL middle-end node 'predecessors' and 'successors'
    Must be very careful to make sure 1) and 2) are consistent

    Insert a csdl expand_uq operation between a variable node and operation node.
    The direction, size of expansions and the numbers of expansions are encoded in expand_strings and num_points.
    Recursively called if multiple expansions are needed
    """

    # 1) -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    op_node = between[1].op
    # remove dependencies of node
    # old_dependencies = node.dependencies.copy()
    # node.remove_dependencies()
    old_dependencies = []
    new_dependencies = []
    dependency = between[0].var

    old_dependencies.append(dependency)
    # Create tile node
    einsum_string = expand_strings[0]

    tile_node = uq_tile(einsum_string=einsum_string)
    tile_node.name = dependency.name + '_' + op_node.name + '_tiling_op'

    # create tiled variable
    tiled_name = dependency.name + '_' + op_node.name + '_tiled'
    tiled_node = Variable(tiled_name)
    new_shape = list(dependency.shape)
    new_shape[0] = new_shape[0]*num_points
    tiled_node.shape = tuple(new_shape)
    # print('klsjdnfkjlnsf', tiled_shape)

    # change dependents of predecessors
    # dependency.remove_dependent_node(op_node) # This does not work for some reason
    dependency.add_dependent_node(tile_node)

    # change dependents/dependencies of tiling node
    tile_node.add_dependency_node(dependency)
    tile_node.add_dependent_node(tiled_node)

    # change dependents/dependencies of tiled node
    tiled_node.add_dependency_node(tile_node)
    tiled_node.add_dependent_node(op_node)

    # change dependencies of original operation
    # node.add_dependency_node(tiled_node)
    new_dependencies.append(tiled_node)

    tile_node.outs = tile_node.dependents
    tiled_node.outs = tile_node.dependents
    dependency.outs = dependency.dependents

    # for old_dependency in old_dependencies:
    #     op_node.remove_dependency_node(old_dependency)

    # for new_dependency in new_dependencies:
    #     op_node.add_dependency_node(new_dependency)
    op_node.dependencies = [new_dependencies[0] if x is old_dependencies[0] else x for x in op_node.dependencies]
    # print(op_node.outs)
    # op_node.outs = op_node.dependents
    # print(op_node.outs)
    # 1) end -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    # 2) -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    graph.remove_edge(*between)
    tile_node_rep = OperationNode(tile_node)
    tiled_node_rep = VariableNode(tiled_node)
    graph.add_edges_from([
        (between[0], tile_node_rep),
        (tile_node_rep,  tiled_node_rep),
        (tiled_node_rep,  between[1]),
    ])

    # 2) end -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    if len(expand_strings) > 1:
        for i in range(len(expand_strings)):
            pass

    # print('EDIT:', between)
    return
