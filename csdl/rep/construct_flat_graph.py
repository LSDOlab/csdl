try:
    from csdl.lang.model import Model
except ImportError:
    pass

from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode

from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.utils.prepend_namespace import prepend_namespace
from typing import Tuple, Set
from networkx import DiGraph, compose, contracted_nodes

from csdl.rep.variable_node import VariableNode
from copy import copy
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
        # select one target node to keep in graph;
        unique_targets[k] = list(tgts)[0]

        # gather out edges from target nodes with same promoted name
        for tgt in tgts:
            fwd_edges.extend(graph.out_edges(tgt))

        # remove reduntant target nodes
        for a, _ in fwd_edges:
            if a not in unique_targets.values():
                graph.remove_node(a)

        # replace edges (u,v) where u is a redundant target node with
        # edges (w,v) where w is target node chosen to be kept in graph
        for _, b in fwd_edges:
            graph.add_edge(unique_targets[k], b)
        graph.add_node(unique_targets[k])

    return unique_targets


def gather_variables_by_promoted_name(
    vars: Dict[str, VariableNode], ) -> Dict[str, VariableNode]:
    """
    Create key value pairs of unique source name to corresponding source
    node
    """
    unique_variables: Dict[str, VariableNode] = dict()
    for k, v in vars.items():
        if k not in unique_variables.keys():
            unique_variables[k] = v
    return unique_variables


def combine_sources_and_targets(
    graph: DiGraph,
    unique_sources: Dict[str, VariableNode],
    unique_targets: Dict[str, VariableNode],
):
    """
    Combine source and target nodes with same promoted names
    """
    nodes = {
        prepend_namespace(s.namespace, s.name): s
        for s in graph.nodes()
    }
    print('sources', unique_sources.keys(), 'targets',
          unique_targets.keys())
    for k, s in unique_sources.items():
        print('names should match:', k,
              prepend_namespace(s.namespace, s.name))
        if k in unique_targets.keys():
            print('should be equal', id(nodes[k]),
                  id(unique_targets[k]))
            out_edges = graph.out_edges(unique_targets[k])
            for _, b in out_edges:
                graph.add_edge(s, b)
            # graph.remove_node(targets[k])


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


def merge_graphs(
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

    # create a flattened copy of the graph for each model node
    for mn in copy(child_model_nodes):
        # child_graph_copy = DiGraph()
        # child_graph_copy.add_edges_from(mn.graph.edges())
        # # also copy ModelNode nodes, which do not have edges
        # child_graph_copy.add_nodes_from(mn.graph.nodes())
        graph = compose(graph, mn.graph)
        graph.remove_node(mn)
        graph = merge_graphs(
            graph,
            namespace=prepend_namespace(namespace, mn.name),
        )

        # all variables in child graph
        child_vars: list[VariableNode] = list(
            filter(lambda x: isinstance(x, VariableNode),
                   mn.graph.nodes()))

        # assign namespace to each variable node
        for v in child_vars:
            if not (mn.promotes is None or v.name in mn.promotes):
                v.namespace = prepend_namespace(
                    namespace, prepend_namespace(v.namespace, mn.name))

    return graph


def merge_nodes_based_on_promotions(graph: DiGraph, ):
    # graph contains all variables in the model hierarchy and no
    # submodels; some target variables are redundant; no variables are
    # merged yet as a result of promotions or connections

    # list of all variables in graph
    vars: list[VariableNode] = list(
        filter(lambda x: isinstance(x, VariableNode), graph.nodes()))

    # map of source promoted names to source node object
    sources: Dict[str, VariableNode] = {
        prepend_namespace(x.namespace, x.name): x
        for x in list(
            filter(lambda x: isinstance(x.var, (Input, Output)), vars))
    }

    # List of all target nodes in graph
    targets: list[VariableNode] = [
        x for x in list(
            filter(lambda x: isinstance(x.var, DeclaredVariable), vars))
    ]
    # Set of all unique target names in graph
    target_names: Set[str] = set(
        [prepend_namespace(x.namespace, x.name) for x in targets])
    print(target_names)

    unique_targets: Dict[str, VariableNode] = dict()
    for name in target_names:
        for target in targets:
            if name not in unique_targets.keys():
                if prepend_namespace(target.namespace,
                                     target.name) == name:
                    unique_targets[name] = target
    print('unique_targets', unique_targets.keys())
    print(unique_targets)
    print('targets',
          [prepend_namespace(x.namespace, x.name) for x in targets])

    # gather all targets and then remove unique targets
    for tgt in unique_targets.values():
        targets.remove(tgt)
    print('targets',
          [prepend_namespace(x.namespace, x.name) for x in targets])

    redundant_targets: Dict[str, Set[VariableNode]] = dict()
    for tgt in targets:
        name = prepend_namespace(tgt.namespace, tgt.name)
        try:
            redundant_targets[name].add(tgt)
        except:
            redundant_targets[name] = {tgt}
    print('redundant_targets', redundant_targets)

    # merge nodes corresponding to locally defined and promoted nodes so
    # that each variable is represented by exactly one node; merge only
    # declared variables; merge nodes only as a result of promotions,
    # not user declared connections
    # for k, tgts in redundant_targets.items():
    #     fwd_edges = []
    #     for tgt in tgts:
    #         print(k, prepend_namespace(tgt.namespace, tgt.name), tgt
    #               in graph.nodes)
    #         fwd_edges.extend(graph.out_edges(tgt))
    #     graph.remove_edges_from(fwd_edges)
    #     for _, op in fwd_edges:
    #         graph.add_edge(unique_targets[k], op)
    for k, tgts in redundant_targets.items():
        for tgt in tgts:
            if k in unique_targets.keys():
                contracted_nodes(
                    graph,
                    unique_targets[k],
                    tgt,
                    self_loops=False,
                    copy=False,
                )

    # # merge nodes from unique sources to unique targets; these are
    # # connections formed automatically by promotions
    for k, src in sources.items():
        if k in unique_targets.keys():
            contracted_nodes(
                graph,
                src,
                unique_targets[k],
                self_loops=False,
                copy=False,
            )


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
    graph = merge_graphs(graph)
    merge_nodes_based_on_promotions(graph)

    # issue connections between nodes due to connections

    # issue_connections_flat_graph(
    #     graph_copy,
    #     connections,
    #     promoted_source_names,
    #     promoted_target_names,
    # )
    print([prepend_namespace(n.namespace, n.name) for n in graph.nodes])

    return graph
