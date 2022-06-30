try:
    from csdl.lang.model import Model
except ImportError:
    pass

from tkinter.font import names
from unicodedata import name
from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.merge_connections import merge_connections
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.rep.model_node import ModelNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.get_nodes import get_model_nodes, get_src_nodes, get_tgt_nodes, get_var_nodes
from csdl.utils.prepend_namespace import prepend_namespace
from typing import Tuple, Set
from networkx import DiGraph, compose, contracted_nodes

from csdl.rep.variable_node import VariableNode
from copy import copy
from collections import Counter
from typing import Dict
from warnings import warn


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
    print('grouped_tgts', grouped_tgts)
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


def merge_graphs(
    graph: DiGraph,
    promoted_to_unpromoted: Dict[str, Set[str]],
    unpromoted_to_promoted: Dict[str, str],
    namespace: str = '',
):
    """
    Copy nodes and edges from graphs in submodels into the graph for the
    main model. User declared promotions and connections are assumed to
    be valid.
    """
    child_model_nodes = get_model_nodes(graph)

    # create a flattened copy of the graph for each model node
    graph.remove_nodes_from(child_model_nodes)
    for mn in copy(child_model_nodes):
        graph = compose(graph, mn.graph)
        graph = merge_graphs(
            graph,
            promoted_to_unpromoted,
            unpromoted_to_promoted,
            namespace=prepend_namespace(namespace, mn.name),
        )

        # all variables in child graph
        child_vars: list[VariableNode] = get_var_nodes(mn.graph)

        # assign namespace to each variable node
        # We are in model A adding model B, promotes = None
        # model B's variables are
        #   (namespace)(name)
        #   ()(x1)
        #   (B.)(x2)
        #   (B.C.)(x3)
        # We promote only ()(x1)

        # AT MODEL C
        # namespace: A.B.C
        # child vars:

        # AT MODEL B
        # child vars: x3
        # namespace: A.B
        # child model names: C
        # new namespace: A.B.C

        # AT MODEL A
        # child vars: x2, x3
        # namespace: A
        # child model names: B
        # new namespaces: B, B.C

        # define full namespace only for unpromoted names;
        # ensure that namespace is only updated for variables that can
        # be promoted

        # unpromoted namespace
        unpromoted_namespace = prepend_namespace(namespace, mn.name)

        for v in child_vars:
            # unpromoted name
            unpromoted_name = prepend_namespace(unpromoted_namespace,
                                                v.name)
            # if variable has not been promoted,
            # variable namespace is unpromoted_namespace.
            # otherwise, variable namespace is the promoted namespace
            if unpromoted_name in promoted_to_unpromoted.keys():
                v.namespace = unpromoted_namespace
            elif unpromoted_name in unpromoted_to_promoted.keys():
                promoted_name = unpromoted_to_promoted[unpromoted_name]
                promoted_namespace = '.'.join(
                    promoted_name.rsplit('.')[:-1])
                v.namespace = promoted_namespace
            elif v.name[0] != '_':
                # promote all automatically named variables
                raise KeyError(f'{unpromoted_name} not found.')

    return graph


def merge_automatically_connected_nodes(graph: DiGraph):
    # graph contains all variables in the model hierarchy and no
    # submodels; some target variables are redundant; no variables are
    # merged yet as a result of promotions or connections

    # list of all variables in graph
    vars: list[VariableNode] = get_var_nodes(graph)

    # map of source promoted names to source node object
    sources: Dict[str, VariableNode] = {
        prepend_namespace(x.namespace, x.name): x
        for x in get_src_nodes(vars)
    }

    # List of all target nodes in graph
    targets: list[VariableNode] = get_tgt_nodes(vars)

    # Set of all unique target names in graph
    target_names: Set[str] = set(
        [prepend_namespace(x.namespace, x.name) for x in targets])

    unique_targets: Dict[str, VariableNode] = dict()
    for name in target_names:
        for target in targets:
            if name not in unique_targets.keys():
                if prepend_namespace(target.namespace,
                                     target.name) == name:
                    unique_targets[name] = target

    # gather all targets and then remove unique targets
    for tgt in unique_targets.values():
        targets.remove(tgt)

    redundant_targets: Dict[str, Set[VariableNode]] = dict()
    for tgt in targets:
        name = prepend_namespace(tgt.namespace, tgt.name)
        try:
            redundant_targets[name].add(tgt)
        except:
            redundant_targets[name] = {tgt}

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


# CONNECTIONS
def validate_connections(
    promoted_to_declared_connections: Dict[Tuple[str, str],
                                           list[Tuple[str, str, str]]],
    sources: Dict[str, VariableNode],
    targets: Dict[str, VariableNode],
):
    # TODO: check that multiple sources are not connected to the same target
    # c = Counter([x for _, x in promoted_to_declared_connections.keys()])
    # for a, b in c:
    #     if c[b] > 1:
    #         msg = "Multiple sources connected to target \'{}\'".format(
    #             b)
    #         for (
    #                 src, tgt
    #         ), connections in promoted_to_declared_connections.items():
    #             if b == tgt:
    #                 for p, q, r in connections:
    #                     msg += "  In model \'{}\', found user declared connection (\'{}\', \'{}\')\n".format(
    #                         r, p, q)

    for (
            promoted_source_candidate, promoted_target_candidate
    ), connections_by_namespace in promoted_to_declared_connections.items(
    ):
        if promoted_source_candidate not in sources:
            msg = "Variable with promoted name \'{}\' is not a valid source for connection.".format(
                promoted_source_candidate)
            for (unpromoted_source_candidate,
                 unpromoted_target_candidate,
                 namespace) in connections_by_namespace:
                msg += "Connection (\'{}\', \'{}\') declared in model \'{}\'".format(
                    unpromoted_source_candidate,
                    unpromoted_target_candidate, namespace)
            raise KeyError()
        if promoted_target_candidate not in targets:
            msg = "Variable with promoted name \'{}\' is not a valid target for connection.".format(
                promoted_target_candidate)
            for (unpromoted_source_candidate,
                 unpromoted_target_candidate,
                 namespace) in connections_by_namespace:
                msg += "Connection (\'{}\', \'{}\') declared in model \'{}\'".format(
                    unpromoted_source_candidate,
                    unpromoted_target_candidate, namespace)
            raise KeyError()


def report_duplicate_connections(
    promoted_to_declared_connections: Dict[Tuple[str, str],
                                           list[Tuple[str, str,
                                                      str]]], ):
    if len(promoted_to_declared_connections) == 0:
        return ''
    duplicates_present = False
    for v in promoted_to_declared_connections.values():
        if len(v) > 1:
            duplicates_present = True
            break

    if duplicates_present is False:
        return ''

    msg = "Duplicate connections found. Each connection is shown using promoted names, followed by duplicate connections as declared by the user.\n"
    for k, v in promoted_to_declared_connections.items():
        if len(v) > 1:
            msg += "\nDuplicate connections found for connection:\n{}\n".format(
                k)
            for a, b, namespace in v:
                msg += "  In model \'{}\', found user declared connection (\'{}\', \'{}\')\n".format(
                    namespace, a, b)

    if len(msg) > 0:
        warn(msg)


def merge_user_connected_nodes(
    graph: DiGraph,
    connections: list[Tuple[str, str]],
    src_nodes_map: Dict[str, VariableNode],
    tgt_nodes_map: Dict[str, VariableNode],
):
    for a, b in connections:
        src_nodes_map[a].tgt_namespace.append(
            tgt_nodes_map[b].namespace)
        src_nodes_map[a].tgt_name.append(tgt_nodes_map[b].name)
        contracted_nodes(
            graph,
            src_nodes_map[a],
            tgt_nodes_map[b],
            self_loops=False,
            copy=False,
        )


def construct_flat_graph(
    graph: DiGraph,
    connections,
    promoted_to_unpromoted,
    unpromoted_to_promoted,
) -> DiGraph:
    graph = merge_graphs(
        graph,
        promoted_to_unpromoted,
        unpromoted_to_promoted,
    )
    merge_automatically_connected_nodes(graph)
    # merge connections within flat graph
    merge_connections(
        graph,
        connections,
        promoted_to_unpromoted,
        unpromoted_to_promoted,
    )

    return graph
