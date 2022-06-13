try:
    from csdl.lang.model import Model
except ImportError:
    pass

from networkx import DiGraph
from csdl.rep.store_abs_unpromoted_names_in_each_node import store_abs_unpromoted_names_in_each_node
from csdl.rep.merge_graphs import merge_graphs
from csdl.rep.store_abs_promoted_names_in_each_node import store_abs_promoted_names_in_each_node
from csdl.rep.merge_promoted_nodes import merge_promoted_nodes
from csdl.rep.merge_connected_nodes import merge_connected_nodes
from copy import copy, deepcopy
from networkx import topological_sort, simple_cycles, DiGraph
from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode


def copy_unflat_graph(graph: DiGraph) -> DiGraph:

    graph_copy: DiGraph = graph.copy()
    model_nodes: list[ModelNode] = list( filter(lambda x: isinstance(x, ModelNode), graph_copy.nodes()))

    for mn in model_nodes:
        mn.graph = copy_unflat_graph(mn.graph)
    return graph_copy


def construct_flat_graph(
    model: 'Model',
    unflat_graph: DiGraph,
) -> DiGraph:
    """
    Create a graph representing the model that contains only Variable
    and Operation nodes. No Subgraph nodes are present in the flattened
    graph. Model namespaces are preserved. All remaining
    `DeclaredVariable` nodes are replaced with `Input` nodes.

    `graph` is a shappw copy, but not a view of the unflat graph; nodes
    are not copied, but edges and references to nodes are copied
    """
    graph: DiGraph = copy_unflat_graph(unflat_graph)
    # gather graphs from submodels and copy them in the main model
    merge_graphs(graph)
    # using unpromoted names, find promoted name for each variable and
    # store that name in the variable node
    store_abs_promoted_names_in_each_node(
        graph,
        model.unpromoted_to_promoted,
    )
    # merge nodes that are connected due to promotions; i.e., find all
    # nodes with same promoted name, delete the declare variable
    # instance, and add new edges connecting source variable to
    # subsequent operations
    merge_promoted_nodes(graph)
    # merge nodes that are connected due to user-specified connections;
    # i.e., find all nodes with validated src/tgt name pairs, delete the
    # declare variable instance, store declared variable instance's name
    # in source variable node, and add new edges connecting source
    # variable to subsequent operations
    merge_connected_nodes(graph, model.connections)

    # check for cycles between variable nodes; the flatened graph
    # contains no model nodes, so any cycles found will beo
    # representations of coupled systems expressed incorrectly by the
    # user
    try:
        _ = topological_sort(graph)
    except:
        cycles = simple_cycles(graph)
        raise ValueError(
            "CSDL found the following cycles in the model: {}\nIf you do not intend to have any coupling within the model, you will need to revise your promotions and connections. If on the other hand, you do want coupling, you will need to use an implicit operation, which is defined in terms of a model that computes a residual value."
            .format(cycles))
    return graph
