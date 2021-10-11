from typing import Optional

import numpy as np

from csdl.utils.gen_hex_name import gen_hex_name
from csdl.utils.slice_to_list import slice_to_list
from csdl.utils.parameters import Parameters

# from csdl.std.binops import (ElementwiseAddition, ElementwiseMultiplication,
# ElementwisePower, ElementwiseSubtraction)


def slice_to_tuple(key: slice, size: int) -> tuple:
    if key.start is None:
        key = slice(0, key.stop, key.step)
    if key.stop is None:
        key = slice(key.start, size, key.step)
    return (key.start, key.stop, key.step)


class Node():
    """
    The ``Node`` class is a base type for nodes in a Directed
    Acyclic Graph (DAG) that represents the computation to be performed
    during model evaluation.
    """

    # A counter for all Node objects created so far
    _count = -1

    def __init__(self, *args, **kwargs):
        Node._count += 1
        _id = gen_hex_name(Node._count)
        self._id = _id
        self.name = _id
        self.dependencies: list = []
        self.dependents: list = []
        self.times_visited = 0
        self._getitem_called = False
        self._decomp = None
        self.indexed_exprs = dict()

        for k, v in kwargs.items():
            if k == 'name':
                self.name = v

    def __iadd__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __iand__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __idiv__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __ifloordiv__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __ilshift__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __imod__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __imul__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __ior__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __ipow__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __irshift__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __isub__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def __ixor__(self, other):
        raise NotImplementedError(
            "In place special methods not implemented for CSDL Node. To modify the value of a variable iteratively, use `Model.implicit_operation`."
        )

    def add_fwd_edges(self):
        for dependency in self.dependencies:
            dependency.add_dependent_node(self)
            dependency.add_fwd_edges()

    def remove_fwd_edges(self):
        """
        Remove fwd edges so that we can update the graph.
        This is *only* used when exposing intermediate variables for a
        composite residual.
        """
        for dependency in self.dependencies:
            dependency.remove_fwd_edges()
        self.dependents = []

    def remove_dependencies(self):
        """
        Remove bwd edges so that we can update the graph.
        This is *only* used when exposing intermediate variables for a
        composite residual.
        """
        for dependency in self.dependencies:
            dependency.remove_dependencies()
        self.dependencies = []

    def add_dependent_node(self, dependent):
        self.dependents.append(dependent)
        self.dependents = list(set(self.dependents))

    def register_nodes(self, nodes: dict):
        """
        Register all nodes in DAG.

        **Parameters**

        nodes: dict[Variable]
            Dictionary of nodes registered so far
        """
        # nodes = set()
        # for r in m.registered_outputs:
        #     nodes.union(gather_nodes(r, nodes))
        # print('gathered nodes:', [node.name for node in list(nodes)])
        # for r in self.registered_outputs:
        #     self.nodes.update(gather_nodes(r, self.nodes))
        # print('gathered nodes:', [node.name for node in list(nodes)])

        for node in self.dependencies:
            # Check for name collisions
            if node._id in nodes.keys():
                if nodes[node._id] is not None:
                    # if name is in use and a separate object is already
                    # registered, then we have a name collision
                    if nodes[node._id] is not node:
                        raise ValueError(
                            "Name collision (", node.name, ") between ",
                            nodes[node._id], " and ", node,
                            "; check that calls to regiser_output and ",
                            "create_* do not give the same name to ",
                            "two outputs")
            else:
                # register node
                nodes[node._id] = node
                node.register_nodes(nodes)

    def incr_times_visited(self):
        """
        Increment number of times a node is visited during ``topological_sort``.
        This is necessary for ``topological_sort`` to determine
        execution order for expressions.
        """
        self.times_visited += 1

    def get_dependency_index(self, candidate) -> Optional[int]:
        """
        Get index of dependency in ``self.dependencies``. Used for
        removing indirect dependencies that woud otherwise affect the
        cost of branches in the DAG, which would affect execution order,
        even with the sme constraints on execution order.

        **Parameters**

        candidate: Variable
            The candidate dependency node

        **Returns**

        Optional[int]
            If ``dependency`` is a dependency of ``self``, then the index of
            ``dependency`` in ``self.dependencies`` is returned. Otherwise,
            ``None`` is returned.
        """
        for index in range(len(self.dependencies)):
            if self.dependencies[index] is candidate:
                return index
        return None

    def remove_dependency_by_index(self, index):
        """
        Remove dependency node, given its index. does nothing if
        ``index`` is out of range. See
        ``Variable.remove_dependency``.

        **Parameters**

        index: int
            Index within ``self.dependencies`` where the node to be
            removed might be
        """
        if index < len(self.dependencies):
            self.dependencies.remove(self.dependencies[index])

    def remove_dependency_node(self, candidate):
        """
        Remove dependency node. Does nothing if `candidate` is not a
        dependency. Used for removing indirect dependencies and
        preventing cycles from forming in DAG.

        **Parameters**

        candidate: Variable
            Node to remove from ``self.dependencies``
        """
        index = self.get_dependency_index(candidate)
        if index is not None:
            self.remove_dependency_by_index(index)

    def get_dependent_index(self, candidate) -> Optional[int]:
        """
        Get index of dependency in ``self.dependencies``. Used for
        removing indirect dependencies that woud otherwise affect the
        cost of branches in the DAG, which would affect execution order,
        even with the sme constraints on execution order.

        **Parameters**

        candidate: Variable
            The candidate dependency node

        **Returns**

        Optional[int]
            If ``dependency`` is a dependency of ``self``, then the index of
            ``dependency`` in ``self.dependencies`` is returned. Otherwise,
            ``None`` is returned.
        """
        for index in range(len(self.dependents)):
            if self.dependents[index] is candidate:
                return index
        return None

    def _dedup_dependencies(self):
        """
        Remove duplicate dependencies. Used when adding a dependency.
        """
        self.dependencies = list(set(self.dependencies))

    def remove_dependent_by_index(self, index):
        """
        Remove dependency node, given its index. does nothing if
        ``index`` is out of range. See
        ``Variable.remove_dependency``.

        **Parameters**

        index: int
            Index within ``self.dependencies`` where the node to be
            removed might be
        """
        if index < len(self.dependents):
            self.dependents.remove(self.dependents[index])

    def remove_dependent_node(self, candidate):
        """
        Remove dependency node. Does nothing if `candidate` is not a
        dependency. Used for removing indirect dependencies and
        preventing cycles from forming in DAG.

        **Parameters**

        candidate: Variable
            Node to remove from ``self.dependencies``
        """
        index = self.get_dependent_index(candidate)
        if index is not None:
            self.remove_dependent_by_index(index)

    def print_dag(self, depth=-1, indent=''):
        """
        Print the graph starting at this node (debugging tool)
        """
        print(indent, id(self), self.name, len(self.dependents),
              self.times_visited, self)
        if len(self.dependencies) == 0:
            print(self.name, 'has no dependencies')
        if depth > 0:
            depth -= 1
        if depth != 0:
            for dependency in self.dependencies:
                dependency.print_dag(depth=depth, indent=indent + ' ')

    def get_num_dependents(self):
        return len(self.dependents)
