from networkx import DiGraph
from typing import Set

class IRNode:

    def __init__(self):
        self.times_visited = 0
        self.name: str = ''
        self.namespace: str = ''

    def incr_times_visited(self):
        self.times_visited += 0

    def compute_path_cost(self, graph: DiGraph):
        """
        Compute the path cost by traversing paths from leaf to this
        node, switching to another path when multiple paths meet at a
        node; cost records traversing nodes that have not yet been
        traversed.
        For example, if there are two paths

        A->B->C->D and E->F->C->D

        then traversal will look like

        A->B->E->F->C->D

        so the cost reflects the cost of all nodes in both paths are
        traverse exactly once
        """
        if self._path_cost_computed == False:
            for p in graph.predecessors(self):
                self._path_cost += p.compute_path_cost(graph)
                # remove cost due to repeated node visits
            for n, times in self._nodes_with_multiple_paths_to_this_node.items(
            ):
                self._path_cost -= (times - 1) * n._path_cost
            self._path_cost_computed = True

        return self._path_cost

    def _get_path_cost(self) -> int:
        """
        Get the path cost from leaf node to this node; for use in
        IRNode.sort_dependencies
        """
        return self._path_cost

    def _name(self):
        return self.name

    def sort_dependencies(self, graph: DiGraph):
        """
        Sort dependencies of each node so that topological sort computes
        an order that traverses all nodes once, from leaf to root,
        starting with the longest path and switching to the next longest
        path prior to continuing past a node where two paths meet
        """
        # sort by cost in descending order will yield an ordering of
        # nodes such that the longest path is always traversed first
        self._sorted_dependencies = sorted(
            graph.predecessors(self),
            key=IRNode._get_path_cost,
            reverse=True,
        )
        # self._sorted_dependencies = graph.predecessors(self)

    def find_nodes(self, graph: DiGraph) -> Set['IRNode']:
        preds: Set[IRNode] = set(graph.predecessors(self))
        for p in preds:
            nodes_with_path_to_this_node = p.find_nodes(graph)
            for n in nodes_with_path_to_this_node:
                if n in self._nodes_with_multiple_paths_to_this_node.keys(
                ):
                    self._nodes_with_multiple_paths_to_this_node[n] += 1
                else:
                    self._nodes_with_multiple_paths_to_this_node[n] = 1
        if graph.out_degree(self) > 1:
            return set(self._nodes_with_multiple_paths_to_this_node.
                       keys()).union({self})
        if len(preds) > 1:
            return set()
        return set(self._nodes_with_multiple_paths_to_this_node.keys())
