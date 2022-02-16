from csdl.core.node import Node
from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.core.subgraph import Subgraph
from csdl.core.operation import Operation
from csdl.core.implicit_operation import ImplicitOperation
from csdl.core.input import Input
from csdl.core.standard_operation import StandardOperation
from csdl.operations.combined import combined
from csdl.utils.check_property import check_property
from typing import List, Union, Dict
from copy import copy
from warnings import warn
import numpy as np


def remove_op_from_dag(op: Operation):
    dependencies = copy(op.dependencies)
    dependents = copy(op.dependents)
    for dep in dependencies:
        dep.remove_dependent_node(op)
    for dep in dependents:
        dep.remove_dependency_node(op)
    op.dependencies = []
    op.dependents = []
    return dependencies, dependents


def insert_op_into_dag(
    op: Operation,
    dependencies: List[Variable],
    dependents: List[Output],
):
    op.dependencies = dependencies
    op.dependents = dependents
    op.outs = tuple(dependents)
    for dep in dependents:
        dep.dependencies = [op]
    for dep in dependencies:
        dep.add_dependent_node(op)


def remove_indirect_dependencies(node: Variable):
    """
    Remove the dependencies that do not constrin execution order. That
    is, if C depends on B and A, and B depends on A, then the execution
    order must be A, B, C, even without the dependence of C on A.

    **Parameters**

    node: Variable
        The node to treat as "root". In ``csdl.model``,
        ``Group._root`` is treated as the "root" node.
    """
    # List of indices corresponding to child node references to remove
    remove: list = []
    for child in node.dependencies:
        for grandchild in child.dependencies:
            index = node.get_dependency_index(grandchild)
            if index is not None:
                remove.append(index)
    # remove duplicate indices
    remove = sorted(list(set(remove)))
    terminal_index = 0
    # children form cycle
    # TODO: explain better
    if len(remove) == len(node.dependencies):
        terminal_index = 1
    for i in reversed(remove):
        if i >= terminal_index:
            node.remove_dependency_by_index(i)


def topological_sort(
        registered_nodes: List[Variable]) -> List[Variable]:
    """
    Perform a topological sort on the Directed Acyclic Graph (DAG).
    If any cycles are detected when traversing the graph,
    ``topological_sort`` will not terminate, and it will cause a memory
    overflow.

    This version of a topological sort is modified so that a node will
    not be added to the sorted list until the node has been visited as
    many times as its in-degree; i.e. the number of dependents.

    **Parameters**

    node: Variable
        The node to treat as "root". In ``csdl.model``,
        ``Group._root`` is treated as the "root" node.

    **Returns**

    list[Variable]
        List of ``Variable`` objects sorted from root to leaf. When
        overriding ``csdl.Model.setup``, the first node will be
        ``Group._root``, and the last will be an ``DocInput``,
        ``Concatenation``, ``ImplicitOutput``, or ``IndepVar``.
    """
    sorted_nodes = []
    stack = registered_nodes
    while stack != []:
        v = stack.pop()
        # Use <= instead of < to ensure that the root node (with zero
        # dependents) is visited; otherwise, no nodes will be added to
        # the list of sorted nodes
        if v.times_visited <= v.get_num_dependents():
            # Iterative Depth First Search (DFS) for a DAG, but node is
            # added to the list of sorted nodes only if all of its parents
            # have been added to the list of sorted nodes;
            # the >= condition ensures that a node with no dependents is
            # never added
            v.incr_times_visited()
            if v.times_visited >= v.get_num_dependents():
                for w in v.dependencies:
                    stack.append(w)

            if v.times_visited == v.get_num_dependents():
                sorted_nodes.append(v)
    return sorted_nodes


from csdl.operations.print_var import print_var


def modified_topological_sort(
    registered_nodes: List[Output],
    subgraphs: List[Subgraph] = [],
) -> List[Node]:
    """
    Perform a topological sort on the Directed Acyclic Graph (DAG).
    If any cycles are detected when traversing the graph,
    ``topological_sort`` will not terminate, and it will cause a memory
    overflow.

    This version of a topological sort is modified so that a node will
    not be added to the sorted list until the node has been visited as
    many times as its in-degree; i.e. the number of dependents.

    **Parameters**

    node: Variable
        The node to treat as "root". In ``csdl.model``,
        ``Group._root`` is treated as the "root" node.

    **Returns**

    list[Variable]
        List of ``Variable`` objects sorted from root to leaf. When
        overriding ``csdl.Model.setup``, the first node will be
        ``Group._root``, and the last will be an ``DocInput``,
        ``Concatenation``, ``ImplicitOutput``, or ``IndepVar``.
    """
    print_operations: List[print_var] = []

    sorted_nodes: List[Union[Node, print_var]] = []
    # set of all nodes with no incoming edge (outputs and subgraphs)
    stack = list(filter(
        lambda x: x.dependents == [], registered_nodes)) + list(
            filter(lambda x: x.dependents == [], subgraphs))
    while stack != []:
        v = stack.pop()
        if v.get_num_dependents() == 0 and isinstance(
                v, Output) and isinstance(v.dependencies[0], print_var):
            # ensure print_var operations are moved to end of model
            print_operations.append(v.dependencies[0])
        elif v.get_num_dependents() == 0:
            # registered outputs that have no dependent nodes
            # KLUDGE: temporary
            # TODO: remove these two lines
            if isinstance(v, Subgraph):
                sorted_nodes.append(v)
            for w in v.dependencies:
                stack.append(w)
        elif v.times_visited < v.get_num_dependents():
            # all other nodes
            v.incr_times_visited()
            if v.times_visited == v.get_num_dependents():
                for w in v.dependencies:
                    stack.append(w)

            if v.times_visited == v.get_num_dependents():
                #     if isinstance(v, Subgraph):
                #         # TODO: raise error
                #         if v in sorted_nodes:
                #             print(
                #                 "Connections made for Model {} forms a cycle"
                #                 .format(v.name))
                #             print("Check the following connections:")
                #             # TODO: these aren't the connections
                #             you're looking for
                #             for a, b in v.submodel.connections:
                #                 print("connect('{}', '{}')".format(a, b))
                #             exit()

                sorted_nodes.append(v)
    # ensure print_var operations are moved to end of model
    sorted_nodes = print_operations + sorted_nodes
    # KLUDGE: there has to be a better way to make sure registered nodes
    # without dependent nodes are sorted
    sorted_nodes = list(
        filter(lambda x: x not in sorted_nodes,
               registered_nodes)) + sorted_nodes
    return sorted_nodes


# def remove_duplicate_nodes(nodes, registered_nodes):
#     from csdl.core.input import Input
#     removed_nodes = []
#     n1_registered = False
#     n2_registered = False
#     for n1 in nodes.values():
#         n1_registered = False
#         for n2 in nodes.values():
#             n2_registered = False
#             if n1 is not n2:
#                 if type(n1) == type(n2):
#                     # if isinstance(n1, (DocInput, Input)) and n1.name != n2.name:
#                     # TODO: ensure that dependencies are note reordered
#                     # TODO: rename inputs and outputs within components
#                     if n1.dependencies == n2.dependencies and len(
#                             n1.dependencies) > 0:
#                         if n1 in registered_nodes:
#                             n1_registered = True
#                         if n2 in registered_nodes:
#                             n2_registered = True
#                         if n2_registered == False and n1 not in removed_nodes:
#                             n1.dependents = list(
#                                 set(n1.dependents).union(n2.dependents))
#                             # swap dependencies
#                             for d in n2.dependents:
#                                 d.remove_dependency_node(n2)
#                                 d.add_dependency_node(n1)
#                             n2.dependents = []
#                             removed_nodes.append(n2)
#                         elif n1_registered == False and n2 not in removed_nodes:
#                             n2.dependents = list(
#                                 set(n1.dependents).union(n2.dependents))
#                             # swap dependencies
#                             for d in n1.dependents:
#                                 d.remove_dependency_node(n1)
#                                 d.add_dependency_node(n2)
#                             n1.dependents = []
#                             removed_nodes.append(n1)
#                         print(
#                             n1.name,
#                             [n.name for n in n1.dependencies],
#                             [n.name for n in n1.dependents],
#                             n2.name,
#                             [n.name for n in n2.dependencies],
#                             [n.name for n in n2.dependents],
#                         )


# some graph theory
def min_in_degree(model) -> int:
    for node in model.sorted_nodes:
        if node not in model.registered_outputs and len(
                node.dependents) < 1:
            return 0
    mid = 1
    for node in filter(lambda x: isinstance(x, Subgraph),
                       model.sorted_nodes):
        mid = min_out_degree(node.submodel)
    return mid


def min_out_degree(model) -> int:
    for node in model.sorted_nodes:
        if not isinstance(node, Input) and len(node.dependencies) < 1:
            return 0
    mod = 1
    for node in filter(lambda x: isinstance(x, Subgraph),
                       model.sorted_nodes):
        mod = min_out_degree(node.submodel)
    return mod


def max_in_degree(model) -> int:
    mid = 0
    for node in model.sorted_nodes:
        mid = max(mid, len(node.dependents))
    for node in filter(lambda x: isinstance(x, Subgraph),
                       model.sorted_nodes):
        mid = max(mid, max_in_degree(node.submodel))
    return mid


def max_out_degree(model) -> int:
    mod = 0
    for node in model.sorted_nodes:
        mod = max(mod, len(node.dependencies))
    for node in filter(lambda x: isinstance(x, Subgraph),
                       model.sorted_nodes):
        mod = max(mod, max_out_degree(node.submodel))
    return mod


def is_tree(model) -> bool:
    """
    check if IR is tree or DAG
    """
    for node in model.sorted_nodes:
        if len(node.dependents) > 1:
            return False
    flag = True
    for node in filter(lambda x: isinstance(x, Subgraph),
                       model.sorted_nodes):
        flag = flag and is_tree(node.submodel)
        if flag is False:
            return flag

    return True


# # TODO: walk graph to find inputs on which outputs depend
# def find_inputs(outputs: List[Output],
#                 output_names: List[str] = None) -> Dict[str, str]:
#     """
#     Find all inputs used to compute each output
#     """
#     inputs = []
#     if output_names is None:
#         l = [x.name for x in outputs]
#         for output_name in l:
#             output = l[0]
#             inputs.append(find_inputs(output.dependencies))
#     else:
#         for output_name in output_names:
#             l = list(filter(lambda x: x.name == output_name, outputs))
#             if len(l) == 0:
#                 raise ValueError()
#         output = l[0]
#         inputs.append(find_inputs(output.dependencies))
#     return inputs

# # TODO: walk graph to find outputs that depend on inputs
# def find_outputs(inputs: List[Output],
#                  input_names: List[str] = None) -> Dict[str, str]:
#     """
#     Find all outputs that depend on each input
#     """
#     pass


def count_std_operations(m, elementwise_only=False) -> int:
    if elementwise_only:
        n = len(
            list(
                filter(
                    lambda x: isinstance(x, StandardOperation) and
                    check_property(x, 'elementwise', True),
                    m.sorted_nodes)))
    else:
        n = len(
            list(
                filter(lambda x: isinstance(x, StandardOperation),
                       m.sorted_nodes)))
    subgraphs = list(
        filter(lambda x: isinstance(x, Subgraph), m.sorted_nodes))
    for sg in subgraphs:
        n += count_std_operations(sg.submodel,
                                  elementwise_only=elementwise_only)
    return n


def count_combined_operations(m) -> int:
    n = len(
        list(filter(lambda x: isinstance(x, combined), m.sorted_nodes)))
    subgraphs = list(
        filter(lambda x: isinstance(x, Subgraph), m.sorted_nodes))
    for sg in subgraphs:
        n += count_combined_operations(sg.submodel)
    return n


def count_operations(m) -> int:
    n = len(
        list(filter(lambda x: isinstance(x, Operation),
                    m.sorted_nodes)))
    subgraphs = list(
        filter(lambda x: isinstance(x, Subgraph), m.sorted_nodes))
    for sg in subgraphs:
        n += count_operations(sg.submodel)
    return n


def count_implicit_operations(m) -> int:
    n = len(
        list(
            filter(lambda x: isinstance(x, ImplicitOperation),
                   m.sorted_nodes)))
    subgraphs = list(
        filter(lambda x: isinstance(x, Subgraph), m.sorted_nodes))
    for sg in subgraphs:
        n += count_implicit_operations(sg.submodel)
    return n


def count_variables(m, vectorized: bool = True) -> int:
    v = list(filter(lambda x: isinstance(x, Variable), m.sorted_nodes))
    if vectorized:
        n = len(v)
    else:
        n = sum([np.prod(x.shape) for x in v])
    subgraphs = list(
        filter(lambda x: isinstance(x, Subgraph), m.sorted_nodes))
    for sg in subgraphs:
        n += count_variables(sg.submodel)
    return n


def count_design_variables(m, vectorized: bool = True) -> int:
    v = list(
        filter(
            lambda x: isinstance(x, Input) and x.name in m.
            design_variables.keys().keys(), m.sorted_nodes))
    if vectorized:
        n = len(v)
    else:
        n = sum([np.prod(x.shape) for x in v])
    subgraphs = list(
        filter(lambda x: isinstance(x, Subgraph), m.sorted_nodes))
    for sg in subgraphs:
        n += count_design_variables(sg.submodel)
    return n


def count_constraints(m, vectorized: bool = True) -> int:
    v = list(
        filter(
            lambda x: isinstance(x, Output) and x in m.
            registered_outputs and x.name in m.constraints.keys(),
            m.sorted_nodes))
    if vectorized:
        n = len(v)
    else:
        n = sum([np.prod(x.shape) for x in v])

    subgraphs = list(
        filter(lambda x: isinstance(x, Subgraph), m.sorted_nodes))
    for sg in subgraphs:
        n += count_constraints(sg.submodel)
    return n


def count_outputs(m, vectorized: bool = True) -> int:
    v = m.registered_outputs
    if vectorized:
        n = len(v)
    else:
        n = sum([np.prod(x.shape) for x in v])
    subgraphs = list(
        filter(lambda x: isinstance(x, Subgraph), m.sorted_nodes))
    for sg in subgraphs:
        n += count_outputs(sg.submodel)
    return n
