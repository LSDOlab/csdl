from typing import List
from csdl.utils.graph import modified_topological_sort_nx
from csdl.rep.ir_node import IRNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.model_node import ModelNode
from csdl.rep.get_registered_outputs_from_graph import get_registered_outputs_from_graph
from csdl.rep.get_model_nodes_from_graph import get_model_nodes_from_graph
from networkx import DiGraph, topological_sort

from csdl.rep.get_nodes import get_input_nodes, get_var_nodes
from csdl.utils.prepend_namespace import prepend_namespace
from networkx import draw_networkx
from networkx import DiGraph, ancestors, simple_cycles


def put_inputs_first(graph: DiGraph, sorted_nodes: List[IRNode]):
    var_nodes = get_var_nodes(graph)
    input_nodes: List[VariableNode] = get_input_nodes(var_nodes)
    sorted_nodes = list(set(input_nodes) -
                        set(sorted_nodes)) + sorted_nodes


def sort_nodes_nx(
    graph: DiGraph,
    namespace: str = '',
    *,
    flat: bool,
) -> List[IRNode]:
    """
    Use Kahn's algorithm to sort nodes, reordering expressions without
    regard for the order in which user registers outputs

    **Returns**

    `List[Node]`
    : nodes in reverse order of execution that guarantees a triangular
    Jacobian structure
    """
    if flat is True:
        try:
            sorted_nodes: List[IRNode] = list(topological_sort(graph))
            put_inputs_first(graph, sorted_nodes)
            return sorted_nodes
        except:
            cycles = list(simple_cycles(graph))
            # TODO: show secondary names of connected variables
            # TODO: multiline error messages
            raise KeyError(
                "Promotions or connections are present that define cyclic relationships between variables. Cycles present are as follows:\n{}\nCycles are shown as lists of unique variable names that form a cycle. To represent cyclic relationships in CSDL, define an implicit operation. See documentation for more details on how to define implicit operations."
                .format([[
                    x.name for x in list(
                        filter(lambda v: isinstance(v, VariableNode),
                               cycle))
                ] for cycle in cycles]))
        # initialize stack for computing topological sort
        stack: List[IRNode] = [
            n for n in graph.nodes if graph.out_degree(n) == 0
        ]
        # find nodes that have multiple paths to later nodes; this
        # is to prevent overcounting the cost to traverse the graph
        # from a leaf node (i.e. input) to another node
        # FIXME: enters infinite loop (?) when implicit operations are
        # present
        # for node in stack:
        #     node.find_nodes(graph)
        # compute path costs to sort later so that topological sort,
        # which does not compute a unique ordering of nodes,
        # computes an ordering so that the longest path from leaf to
        # root is traversed, followed by the next longest path, and
        # so on; this is to ensure that variables may be allocated
        # as late as possible during program execution
        for node in stack:
            node.compute_path_cost(graph)
        # sort the dependencies of each node by cost in descending
        # order to ensure topological sort computes ordering
        # described above
        for node in graph.nodes:
            node.sort_dependencies(graph)
        # we have set up the graph so that of all possible
        # topological sorts, now we can compute the desired
        # topological sort
        sorted_nodes: List[IRNode] = modified_topological_sort(
            graph, stack)
        put_inputs_first(graph, sorted_nodes)
        put_outputs_last(graph, sorted_nodes)

        return sorted_nodes
    else:
        model_nodes: List[ModelNode] = get_model_nodes_from_graph(graph)
        for m in model_nodes:
            m.sorted_nodes = sort_nodes_nx(
                DiGraph(m.graph),
                namespace=m.namespace,
                flat=False,
            )
        try:
            sorted_nodes: List[IRNode] = list(topological_sort(graph))
            put_inputs_first(graph, sorted_nodes)
            return sorted_nodes
        except:
            sorted_nodes: List[IRNode] = list(graph.nodes)
            return sorted_nodes
