try:
    from csdl.lang.model import Model
except ImportError:
    pass

from networkx import DiGraph
from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode

from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.utils.prepend_namespace import prepend_namespace
from typing import Tuple, Set
from networkx import DiGraph

from csdl.rep.variable_node import VariableNode
from typing import Dict


def gather_targets_by_promoted_name(
    ungrouped_tgts: Dict[str, VariableNode],
    promotes: Set[str] | None,
    namespace: str,
) -> Dict[str, Set[VariableNode]]:
    """
    Create key value pairs of target name to all target nodes with same
    name; resulting dictionary will be used to eliminate redundant
    edges/nodes in flattened graph
    """
    grouped_tgts: Dict[str, Set[VariableNode]] = dict()
    for k, v in ungrouped_tgts.items():
        if promotes is None or k in promotes:
            try:
                grouped_tgts[k].add(v)
            except:
                grouped_tgts[k] = {v}
        else:
            v.namespace = namespace
            name = prepend_namespace(namespace, k)
            try:
                grouped_tgts[name].add(v)
            except:
                grouped_tgts[name] = {v}
    return grouped_tgts


def isolate_unique_targets(
    graph: DiGraph,
    grouped_tgts: Dict[str, Set[VariableNode]],
) -> Dict[str, VariableNode]:
    unique_targets: Dict[str, VariableNode] = dict()
    # TODO: what type?
    fwd_edges = []
    for k, tgts, in grouped_tgts.items():
        # remove edges containing redundant target nodes
        for tgt in tgts:
            fwd_edges.extend(graph.out_edges(tgt))
        graph.remove_edges_from(fwd_edges)

        # select one target node to keep in graph;
        # use a loop since we can't use a zero index in a set and
        # because all redundant target nodes are identical
        for tgt in tgts:
            unique_targets[k] = tgt
            break

        # remove reduntant target nodes
        graph.remove_nodes_from([a for a, _ in fwd_edges])

        # replace edges (u,v) where u is a redundant target node with
        # edges (w,v) where w is target node chosen to be kept in graph
        for _, b in fwd_edges:
            graph.add_edge(unique_targets[k], b)

    return unique_targets


def gather_sources_by_promoted_name(
    srcs: Dict[str, VariableNode],
    promotes: Set[str] | None,
    namespace: str,
) -> Dict[str, VariableNode]:
    """
    Create key value pairs of unique source name to corresponding source
    node
    """
    unique_sources: Dict[str, VariableNode] = dict()
    for k, v in srcs.items():
        if promotes is None or k in promotes:
            unique_sources[k] = v
        else:
            v.namespace = namespace
            name = prepend_namespace(namespace, k)
            unique_sources[name] = v
    return unique_sources


def combine_sources_targets(
    graph: DiGraph,
    sources: Dict[str, VariableNode],
    targets: Dict[str, VariableNode],
):
    """
    Combine source and target nodes with same promoted names
    """
    print('sources', sources.keys(), 'targets', targets.keys())
    for k, s in sources.items():
        # TODO: type?
        if k in targets.keys():
            out_edges = list(graph.out_edges(targets[k]))
            # TODO: how to iterate over edges?
            for _, b in out_edges:
                graph.add_edge(s, b)
            graph.remove_node(targets[k])


def combine_connected_nodes(
    graph: DiGraph,
    source: VariableNode,
    tgt: VariableNode,
):
    """
    Combine source and target nodes with same promoted names
    """
    # TODO: type?
    out_edges = list(graph.out_edges(tgt))
    # TODO: how to iterate over edges?
    for q in out_edges:
        graph.add_edge(source, b)
    graph.remove_node(tgt)


def merge_graphs_based_on_promotions(
    graph: DiGraph,
    namespace: str = '',
):
    """
    Copy nodes and edges from graphs in submodels into the graph for the
    main model. User declared promotions and connections are assumed to
    be valid.
    """
    child_model_nodes: list[ModelNode] = list(
        filter(lambda x: isinstance(x, ModelNode), graph.nodes()))

    unique_sources: Dict[str, VariableNode] = dict()
    unique_targets: Dict[str, VariableNode] = dict()
    # create a flattened copy of the graph for each model node
    for mn in child_model_nodes:
        child_graph_copy = DiGraph(mn.graph)
        merge_graphs_based_on_promotions(
            child_graph_copy,
            namespace=prepend_namespace(namespace, mn.name),
        )
        graph.add_edges_from(child_graph_copy.edges())

        # merge nodes corresponding to locally defined and promoted
        # nodes so that each variable is represented by exactly one
        # node; merge nodes only as a result of promotions, not user
        # declared connections
        child_vars: list[VariableNode] = list(
            filter(lambda x: isinstance(x, VariableNode),
                   child_graph_copy.nodes()))

        # merge targets from child into unique targets
        child_ungrouped_tgts: Dict[str, VariableNode] = {
            x.name: x
            for x in list(
                filter(lambda x: isinstance(x.var, DeclaredVariable),
                       child_vars))
        }
        child_grouped_tgts = gather_targets_by_promoted_name(
            child_ungrouped_tgts,
            None if mn.promotes is None else set(mn.promotes),
            prepend_namespace(namespace, mn.name),
        )
        unique_targets.update(isolate_unique_targets(
            child_graph_copy,
            child_grouped_tgts,
        ))

        # find sources by promoted (unique) name
        child_srcs: Dict[str, VariableNode] = {
            x.name: x
            for x in list(
                filter(lambda x: isinstance(x.var, (Input, Output)),
                       child_vars))
        }
        # FIXME: not finding sources in current level
        unique_sources.update(gather_sources_by_promoted_name(
            child_srcs,
            None if mn.promotes is None else set(mn.promotes),
            prepend_namespace(namespace, mn.name),
        ))
        # FIXME: not combining targets
        combine_sources_targets(
            child_graph_copy,
            unique_sources,
            unique_targets,
        )
    graph.remove_nodes_from(child_model_nodes)


## CONNECTIONS


def issue_connections_flat_graph(
    flat_graph: DiGraph,
    connections: list[Tuple[str, str]],
    promoted_source_names: Set[str],
    promoted_target_names: Set[str],
):
    vars: list[VariableNode] = list(
        filter(lambda x: isinstance(x, VariableNode),
               flat_graph.nodes()))
    src_nodes: list[VariableNode] = list(
        filter(
            lambda x: prepend_namespace(x.namespace, x.name) in
            promoted_source_names, vars))
    tgt_nodes: list[VariableNode] = list(
        filter(
            lambda x: prepend_namespace(x.namespace, x.name) in
            promoted_target_names, vars))
    src_nodes_map: Dict[str, VariableNode] = {
        prepend_namespace(x.namespace, x.name): x
        for x in src_nodes
    }
    tgt_nodes_map: Dict[str, VariableNode] = {
        prepend_namespace(x.namespace, x.name): x
        for x in tgt_nodes
    }
    for a, b in connections:
        combine_connected_nodes(
            flat_graph,
            src_nodes_map[a],
            tgt_nodes_map[b],
        )


def construct_flat_graph(
    graph: DiGraph,
    connections: list[Tuple[str, str]],
    promoted_source_names: Set[str],
    promoted_target_names: Set[str],
) -> DiGraph:
    graph_copy = DiGraph(graph)
    merge_graphs_based_on_promotions(graph_copy)
    # issue_connections_flat_graph(
    #     graph_copy,
    #     connections,
    #     promoted_source_names,
    #     promoted_target_names,
    # )
    return graph_copy
