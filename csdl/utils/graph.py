from csdl.core.variable import Variable
from typing import List
from copy import copy
from warnings import warn


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


def paper_topological_sort(
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
    stack = copy(registered_nodes)
    while stack != []:
        v = stack.pop()
        if v.get_num_dependents() == 0:
            sorted_nodes.append(v)
        elif v.times_visited < v.get_num_dependents():
            v.incr_times_visited()

            # modify to respect order that user registered outputs
            if v in registered_nodes:
                if not (all(x in sorted_nodes for x in v.dependents)):
                    raise Warning('{} is registered late'.format(
                        v.name))

            if v.times_visited == v.get_num_dependents():
                for w in v.dependencies:
                    stack.append(w)

            if v.times_visited == v.get_num_dependents():
                sorted_nodes.append(v)
    return sorted_nodes


from csdl.operations.print_var import print_var


def modified_topological_sort(
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
    print_operations = []
    sorted_nodes = []
    stack = copy(registered_nodes)
    while stack != []:
        v = stack.pop()
        if v.get_num_dependents() == 0:
            if isinstance(v, Variable) and isinstance(
                    v.dependencies[0], print_var):
                # ensure print_var operations are moved to end of model
                print_operations.append(v.dependencies[0])
            else:
                # registered outputs that have no dependent nodes
                sorted_nodes.append(v)
                for w in v.dependencies:
                    stack.append(w)
        elif v.times_visited < v.get_num_dependents():
            # all other nodes
            v.incr_times_visited()

            # modify to respect order that user registered outputs
            if v in registered_nodes:
                if not (all((x in sorted_nodes for x in v.dependents))):
                    # TODO: tell user which model this is
                    # NOTE: raising a warning would terminate the
                    # program here
                    warn(
                        "{} is registered late."
                        "This will result in unnecessary feedback in "
                        "your model, and will require an iterative "
                        "solver to finish evaluating the model.".format(
                            v.name), )

            if v.times_visited == v.get_num_dependents():
                for w in v.dependencies:
                    stack.append(w)

            if v.times_visited == v.get_num_dependents():
                sorted_nodes.append(v)
    # ensure print_var operations are moved to end of model
    sorted_nodes = print_operations + sorted_nodes
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
