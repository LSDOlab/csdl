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
from csdl.lang.custom_operation import CustomOperation
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.rep.model_node import ModelNode
from csdl.rep.get_nodes import get_model_nodes, get_src_nodes, get_tgt_nodes, get_var_nodes
from csdl.utils.prepend_namespace import prepend_namespace
from typing import Tuple, Set
from networkx import DiGraph, compose, contracted_nodes

from copy import copy
from collections import Counter
from typing import Dict, Union, List
from warnings import warn


class GraphWithMetadata:

    def __init__(self, graph: DiGraph):
        self.graph = graph
        self.promoted_to_node = dict()
        self.unpromoted_to_node = dict()
        self.connected_tgt_nodes_to_source_nodes = dict()


def gather_targets_by_promoted_name(
    ungrouped_tgts: Dict[str, VariableNode],
    promotes: Union[Set[str], None],
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
    hierarchy: List[str],
    namespace: str = '',
    analytics_filename = None,
):
    """
    Copy nodes and edges from graphs in submodels into the graph for the
    main model. User declared promotions and connections are assumed to
    be valid.
    """
    # OLD:
    # child_model_nodes = get_model_nodes(graph)

    # NEW:
    child_model_nodes = graph.model_nodes

    # create a flattened copy of the graph for each model node
    graph.remove_nodes_from(child_model_nodes)
    graph.model_nodes = set()
    for mn in copy(child_model_nodes):

        # compose is slower than manually adding edges
        # graph = compose(graph, mn.graph)
        graph.add_edges_from(mn.graph.edges())
        graph.add_nodes_from(mn.graph.nodes())
        graph.model_nodes = mn.graph.model_nodes

        # if analytics_filename is not None:
        #     len_hierarchy = len(hierarchy)
        #     prepend = '	'*len_hierarchy
        #     prepend_model =  prepend+ '-'
        #     prepend_var = prepend+'	' + '*'
        #     with open(analytics_filename, 'a') as f:
        #         f.write(f'{prepend_model}{mn.name}\n')
        #         for node in mn.graph.nodes:
        #             if isinstance(node, VariableNode):
        #                 if node.name == node.var._id:
        #                     continue
        #                 f.write(f'{prepend_var}{hierarchy + [mn.name]}{prepend_namespace(node.namespace, node.name)}\n')

        # graph = merge_graphs(
        #     graph,
        #     promoted_to_unpromoted,
        #     unpromoted_to_promoted,
        #     hierarchy = hierarchy + [mn.name],
        #     namespace=prepend_namespace(namespace, mn.name),
        #     analytics_filename = analytics_filename,
        # )

        # all variables in child graph
        child_vars: List[VariableNode] = get_var_nodes(mn.graph)

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
            v.unpromoted_namespace = unpromoted_namespace

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
            # elif v.name[0] != '_':
            elif v.name != v.var._id:
                # promote all automatically named variables
                # raise KeyError(f'{unpromoted_name} not found.')
                pred_list = list(graph.predecessors(v))
                if len(pred_list) == 0:
                    raise KeyError(f'{unpromoted_name} not found. Currently processing model {mn.name}.')

                previous_op = pred_list[0].op
                if not isinstance(previous_op, CustomOperation):
                    raise KeyError(f'{unpromoted_name} not found. Currently processing model {mn.name}.')
        
        if analytics_filename is not None:
            write_hierarchy_info_to_file(
                analytics_filename,
                hierarchy,
                mn.name,
                mn.graph,
            )
        # if analytics_filename is not None:
        #     h_w_model = hierarchy + [mn.name]
        #     len_hierarchy = len(hierarchy)
        #     spaces = '   '
        #     promote_marker = '---'
        #     promote_marker_f = '|--'

        #     prepend = spaces*len_hierarchy
        #     prepend_model =  prepend+ '| '
        #     postpend_model =  ''
        #     prepend_var_full = prepend+spaces + '*'
        #     with open(analytics_filename, 'a') as f:
        #         f.write(f'{prepend_model}{mn.name}{postpend_model}\n')
        #         for node in mn.graph.nodes:
        #             if isinstance(node, VariableNode):
        #                 if node.name == node.var._id:
        #                     continue
                        
        #                 # OLD:
        #                 # split_namespace = ['']+ node.namespace.split('.')
        #                 # prepend_var = ''
        #                 # first_p_model = False
        #                 # for model_level, model_name in enumerate(h_w_model):
                        
        #                 #     if model_level >= len(split_namespace)-1:
        #                 #         if not first_p_model:
        #                 #             # promote_marker_f = promote_marker
        #                 #             # promote_marker_f[0] = '|'
        #                 #             prepend_var += promote_marker_f
        #                 #         else:
        #                 #             prepend_var += promote_marker
        #                 #         first_p_model = True
        #                 #         continue
        #                 #     else:
        #                 #         if split_namespace[model_level] != model_name:
        #                 #             raise ValueError(f'Namespace mismatch: {split_namespace[model_level]} != {model_name}')
        #                 #         else:
        #                 #             prepend_var += spaces

        #                 split_namespace = ['']+ node.namespace.split('.')
        #                 postpend_var = ''
        #                 num_pros = len_hierarchy
        #                 for model_level, model_name in enumerate(h_w_model):
        #                     if model_level >= len(split_namespace)-1:
        #                         postpend_var += str(num_pros)
        #                     else:
        #                         if split_namespace[model_level] != model_name:
        #                             raise ValueError(f'Namespace mismatch: {split_namespace[model_level]} != {model_name}')
        #                         else:
        #                             postpend_var += '.'
        #                     num_pros-=1

        #                 if isinstance(node.var, DeclaredVariable):
        #                     var_type = 'dec'
        #                 elif isinstance(node.var, Input):
        #                     var_type = 'in '
        #                 elif isinstance(node.var, Output):
        #                     var_type = 'out'
        #                 else:
        #                     raise ValueError(f'Unknown variable type: {node.var}')
        #                 # f.write(f'{prepend_var}*{node.name}\n')
        #                 # main_string = f'{prepend_var_full}{node.name}'
        #                 main_string = f'{prepend_var_full}{node.name}'
        #                 pad_string1 = ' '*(max(90 - len(main_string), 5))+ '  '

        #                 main_string2 = f'{pad_string1}{var_type}/{postpend_var}'
        #                 pad_string2 = ' '*(max(120 - len(main_string2)-len(main_string), 5))+ '  '

        #                 main_string3 = f'{pad_string2}p:{prepend_namespace(node.namespace, node.name)} \t(up:{prepend_namespace(node.unpromoted_namespace, node.name)})'
        #                 f.write(main_string)
        #                 f.write(main_string2)
        #                 f.write(f'{main_string3}\n')
        #                 # f.write(f'{pad_string1}{var_type}/{postpend_var}{prepend_namespace(node.namespace, node.name)} \t(unpromoted:   {prepend_namespace(node.unpromoted_namespace, node.name)})\n')
                        
        #                 # temp = prepend+spaces + '    '
        #                 # f.write(f'{temp}{var_type}\n')

        #                 # prepend_var = prepend+spaces + '*'
        #                 # f.write(f'{prepend_var}{h_w_model}{prepend_namespace(node.namespace, node.name)}\n')
                        

        #                 # split_namespace = node.namespace.split('.')
        #                 # if split_namespace == ['']:
        #                 #     continue
                        
        #                 # h_level = len(split_namespace)
        #                 # # split_namespace = split_namespace[1:]

        #                 # if h_w_model[0:h_level] != split_namespace:
        #                 #     raise ValueError(f'Namespace mismatch: {h_w_model[0:h_level]} != {split_namespace}')
        #                 # else:
        #                 #     print(h_w_model[0:h_level],split_namespace)
        
        graph = merge_graphs(
            graph,
            promoted_to_unpromoted,
            unpromoted_to_promoted,
            hierarchy = hierarchy + [mn.name],
            namespace=prepend_namespace(namespace, mn.name),
            analytics_filename = analytics_filename,
        )
    return graph


def merge_automatically_connected_nodes(graph: DiGraph):
    # graph contains all variables in the model hierarchy and no
    # submodels; some target variables are redundant; no variables are
    # merged yet as a result of promotions or connections

    # list of all variables in graph
    vars: List[VariableNode] = get_var_nodes(graph)

    # map of source promoted names to source node object
    sources: Dict[str, VariableNode] = {
        prepend_namespace(x.namespace, x.name): x
        for x in get_src_nodes(vars)
    }

    # List of all target nodes in graph
    targets: List[VariableNode] = get_tgt_nodes(vars)

    # Set of all unique target names in graph
    target_names: Set[str] = set(
        [prepend_namespace(x.namespace, x.name) for x in targets])

    unique_targets: Dict[str, VariableNode] = dict()

    # Victors old code
    # for name in target_names:
    #     for target in targets:
    #         if name not in unique_targets.keys():
    #             if prepend_namespace(target.namespace,
    #                                  target.name) == name:
    #                 unique_targets[name] = target
    #                 call += 1

    # TC2 updated code
    # import time
    # start = time.time()
    # for target in targets:
    #     target.tgt_prepended = prepend_namespace(target.namespace, target.name)
    # for name in target_names:
    #     for target in targets:
    #         # tgt_prepended = prepend_namespace(target.namespace, target.name)
    #         if target.tgt_prepended == name:
    #             unique_targets[name] = target
    #             break
    # print(f'TIME ({len(target_names)} x {len(target_names)}):', time.time() - start)

    # v0.1 updated code
    for target in targets:
        target.tgt_prepended = prepend_namespace(target.namespace, target.name)

        # Uncomment to check if declared variable values are being overwritten
        # import numpy as np
        # if target.tgt_prepended in unique_targets:
        #     if np.linalg.norm(target.var.val) != np.linalg.norm(unique_targets[target.tgt_prepended].var.val):
        #         print('WARNING:',target.tgt_prepended, target.var.val, unique_targets[target.tgt_prepended].var.val)
            # raise ValueError('Error?')

        if target.tgt_prepended not in unique_targets.keys():
            unique_targets[target.tgt_prepended] = target
        else:
            num_models_from_root_new = target.unpromoted_namespace.count('.')
            num_models_from_root_current = unique_targets[target.tgt_prepended].unpromoted_namespace.count('.')

            # For variables merged from promotions, we use the variable node defined at model closest to the root
            if num_models_from_root_new < num_models_from_root_current:
                unique_targets[target.tgt_prepended] = target
            elif num_models_from_root_current == num_models_from_root_new:
                if target.var.default_val == False:
                    unique_targets[target.tgt_prepended] = target

    # gather all targets and then remove unique targets
    for tgt in unique_targets.values():
        targets.remove(tgt)

    redundant_targets: Dict[str, Set[VariableNode]] = dict()
    for tgt in targets:
        name = tgt.tgt_prepended

        # Instead of try/except
        if name not in redundant_targets.keys():
            redundant_targets[name] = {tgt}
        else:
            redundant_targets[name].add(tgt)
        
        # OLD:
        # try:
        #     redundant_targets[name].add(tgt)
        # except:
        #     redundant_targets[name] = {tgt}

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
                unique_targets[k].declared_to.add(tgt)
                contracted_nodes(
                    graph,
                    unique_targets[k],
                    tgt,
                    self_loops=False,
                    copy=False,
                )
                # tgt.var.rep_node = unique_targets[k]
                tgt.var.add_IR_mapping(unique_targets[k])

    # # merge nodes from unique sources to unique targets; these are
    # # connections formed automatically by promotions
    for k, src in sources.items():
        if k in unique_targets.keys():
            src.declared_to.add(unique_targets[k])
            src.declared_to.update(unique_targets[k].declared_to)
            contracted_nodes(
                graph,
                src,
                unique_targets[k],
                self_loops=False,
                copy=False,
            )
            # unique_targets[k].var.rep_node = src
            unique_targets[k].var.add_IR_mapping(src)


# CONNECTIONS
def validate_connections(
    promoted_to_declared_connections: Dict[Tuple[str, str],
                                           List[Tuple[str, str, str]]],
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
                                           List[Tuple[str, str,
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
    connections: List[Tuple[str, str]],
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
        # tgt_nodes_map[b].var.rep_node = src_nodes_map[a]
        tgt_nodes_map[b].var.add_IR_mapping(src_nodes_map[a])


def construct_flat_graph(
    graph: DiGraph,
    connections,
    promoted_to_unpromoted,
    unpromoted_to_promoted,
    analytics: bool = False,
    rep_name = ''
) -> GraphWithMetadata:
    
    if analytics:
        # Analytics directory
        import os
        name_prepend = ''
        if rep_name != '':
            name_prepend = f'_{rep_name}'
        directory_name = f'MODEL_SUMMARY{name_prepend}'
        model_summary_filename = f'{directory_name}/HIERARCHY.txt'
        model_guide_filename = f'{directory_name}/guide.txt'

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        with open(model_guide_filename, 'w') as f:
            f.write("""
HIERARCHY.txt outlines the defined model hierarchy and all defined variables in the model.
Each row is a variable (*) or model (|). If a variable is created in a model, it is indented under that model and lists (in order to the right)
the variable type (declared/registered output/created input), the promotion level, the shape, the promoted name, and the unpromoted name.

example:

    # Base model
    model = csdl.Model()

    # First level model
    model1 = csdl.Model()
    model.add(model1, 'ModelA', promotes = [])

    # Second level model
    model2 = csdl.Model()
    model2.create_input('x', val=3)
    model1.add(model2, 'ModelB', promotes = ['x'])
    # model1.declare_variable('x')

    # declare variable
    x = model.declare_variable('ModelA.x')
    model.register_output('y', x*2)

looks like:
| <SYSTEM LEVEL>
   *y                                                                                       out/./(1,)                         p:y 	(up:y)
   *ModelA.x                                                                                dec/./(1,)                         p:ModelA.x 	(up:ModelA.x)
   | ModelA
      | ModelB
         *x                                                                                 in /.10/(1,)                       p:ModelA.x 	(up:ModelA.ModelB.x)


We can see that the promoted name "ModelA.x" (right of 'p:') is listed twice, meaning they get promoted to the same variable.""")

        with open(model_summary_filename, 'w') as f:
            f.write('')
        


        # mn_temp = 
        write_hierarchy_info_to_file(
            model_summary_filename,
            None,
            '<SYSTEM LEVEL>',
            graph,
        )
    else:
        model_summary_filename = None

    graph = merge_graphs(
        graph,
        promoted_to_unpromoted,
        unpromoted_to_promoted,
        hierarchy = [''],
        analytics_filename = model_summary_filename,
    )

    merge_automatically_connected_nodes(graph)

    graph_meta = merge_connections(
        GraphWithMetadata(graph),
        connections,
        promoted_to_unpromoted,
        unpromoted_to_promoted,
    )

    return graph_meta


def write_hierarchy_info_to_file(
        fname,
        hierarchy,
        mn_name,
        mn_graph,
    ):

    if hierarchy == None:
        h_w_model = ['']
        len_hierarchy = 0
    else:
        h_w_model = hierarchy + [mn_name]
        len_hierarchy = len(hierarchy)
    spaces = '   '
    promote_marker = '---'
    promote_marker_f = '|--'

    prepend = spaces*len_hierarchy
    prepend_model =  prepend+ '| '
    postpend_model =  ''
    prepend_var_full = prepend+spaces + '*'
    with open(fname, 'a') as f:
        f.write(f'{prepend_model}{mn_name}{postpend_model}\n')
        for node in mn_graph.nodes:
            if isinstance(node, VariableNode):
                if node.name == node.var._id:
                    continue
                
                # OLD:
                # split_namespace = ['']+ node.namespace.split('.')
                # prepend_var = ''
                # first_p_model = False
                # for model_level, model_name in enumerate(h_w_model):
                
                #     if model_level >= len(split_namespace)-1:
                #         if not first_p_model:
                #             # promote_marker_f = promote_marker
                #             # promote_marker_f[0] = '|'
                #             prepend_var += promote_marker_f
                #         else:
                #             prepend_var += promote_marker
                #         first_p_model = True
                #         continue
                #     else:
                #         if split_namespace[model_level] != model_name:
                #             raise ValueError(f'Namespace mismatch: {split_namespace[model_level]} != {model_name}')
                #         else:
                #             prepend_var += spaces

                split_namespace = ['']+ node.namespace.split('.')
                postpend_var = ''
                num_pros = len_hierarchy
                for model_level, model_name in enumerate(h_w_model):
                    if model_level >= len(split_namespace)-1:
                        postpend_var += str(num_pros)
                    else:
                        if split_namespace[model_level] != model_name:
                            raise ValueError(f'Namespace mismatch: {split_namespace[model_level]} != {model_name}')
                        else:
                            postpend_var += '.'
                    num_pros-=1

                if isinstance(node.var, DeclaredVariable):
                    var_type = 'dec'
                elif isinstance(node.var, Input):
                    var_type = 'in '
                elif isinstance(node.var, Output):
                    var_type = 'out'
                else:
                    raise ValueError(f'Unknown variable type: {node.var}')
                # f.write(f'{prepend_var}*{node.name}\n')
                # main_string = f'{prepend_var_full}{node.name}'
                main_string = f'{prepend_var_full}{node.name}'
                pad_string1 = ' '*(max(90 - len(main_string), 5))+ '  '

                main_string2 = f'{pad_string1}{var_type}/{postpend_var}/{node.var.shape}'
                pad_string2 = ' '*(max(125 - len(main_string2)-len(main_string), 5))+ '  '

                main_string3 = f'{pad_string2}p:{prepend_namespace(node.namespace, node.name)} \t(up:{prepend_namespace(node.unpromoted_namespace, node.name)})'
                f.write(main_string)
                f.write(main_string2)
                f.write(f'{main_string3}\n')
                # f.write(f'{pad_string1}{var_type}/{postpend_var}{prepend_namespace(node.namespace, node.name)} \t(unpromoted:   {prepend_namespace(node.unpromoted_namespace, node.name)})\n')
                
                # temp = prepend+spaces + '    '
                # f.write(f'{temp}{var_type}\n')

                # prepend_var = prepend+spaces + '*'
                # f.write(f'{prepend_var}{h_w_model}{prepend_namespace(node.namespace, node.name)}\n')
                

                # split_namespace = node.namespace.split('.')
                # if split_namespace == ['']:
                #     continue
                
                # h_level = len(split_namespace)
                # # split_namespace = split_namespace[1:]

                # if h_w_model[0:h_level] != split_namespace:
                #     raise ValueError(f'Namespace mismatch: {h_w_model[0:h_level]} != {split_namespace}')
                # else:
                #     print(h_w_model[0:h_level],split_namespace)
