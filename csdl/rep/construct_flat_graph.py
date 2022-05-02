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
from copy import copy


def construct_flat_graph(
    model: 'Model',
    unflat_graph: DiGraph,
) -> DiGraph:
    """
    Create a graph representing the model that contains only Variable
    and Operation nodes. No Subgraph nodes are present in the flattened
    graph. Model namespaces are preserved. All remaining
    `DeclaredVariable` nodes are replaced with `Input` nodes.

    `graph` is a deepcopy of the unflat graph
    """
    graph = copy(unflat_graph)
    store_abs_unpromoted_names_in_each_node(graph)
    merge_graphs(graph)
    store_abs_promoted_names_in_each_node(
        graph,
        model.unpromoted_to_promoted,
    )
    merge_promoted_nodes(graph)
    merge_connected_nodes(graph, model.connections)
    return graph
