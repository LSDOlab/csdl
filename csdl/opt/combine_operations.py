from csdl.lang.output import Output
from csdl.lang.operation import Operation
from csdl.lang.standard_operation import StandardOperation
from csdl.operations.combined import combined
from csdl.rep.graph_representation import GraphRepresentation
from csdl.rep.ir_node import IRNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.variable_node import VariableNode
from networkx import DiGraph
from typing import Set
from csdl.utils.check_property import check_property
from csdl.rep.get_registered_outputs_from_graph import get_registered_outputs_from_graph
from csdl.rep.get_nodes import get_operation_nodes
from csdl.rep.get_model_nodes_from_graph import get_model_nodes_from_graph
from csdl.rep.graph_representation import GraphRepresentation
from csdl.rep.apply_fn_to_implicit_operation_nodes import apply_fn_to_implicit_operation_nodes
from copy import deepcopy


def combinable(
    graph: DiGraph,
    registered_outputs: Set[VariableNode],
    op: OperationNode,
):
    preds = list(graph.predecessors(op))
    if len(preds) != 1 or len(list(graph.successors(op))) != 1:
        return False
    if list(preds)[0] in registered_outputs:
        return False
    if isinstance(op.op, StandardOperation):
        return check_property(op.op, 'elementwise', True)
    return False


def insert_new_op(
    graph: DiGraph,
    v01: VariableNode,
    op1: OperationNode,
    v12: VariableNode,
    op2: OperationNode,
    v23: VariableNode,
):
    new_op = combined()
    # use isinstance here only for strict type checking
    if isinstance(op1.op, StandardOperation):
        # op1.op.define_compute_strings()
        new_op.compute_string += op1.op.compute_string + '\n'
    if isinstance(op2.op, StandardOperation):
        # op2.op.define_compute_strings()
        new_op.compute_string += op2.op.compute_string + '\n'
    if isinstance(v23.var, Output):
        new_op.outs = (v23.var, )
        new_op.dependents = [v23.var]
    c = OperationNode(new_op)

    # KLUDGE: storing front end nodes in
    # GraphRepresentation nodes until future update
    op1.op.dependencies = []
    op2.op.dependencies = []
    op1.op.dependents = []
    op2.op.dependents = []
    op1.op.outs = ()
    op2.op.outs = ()
    v23.var.dependencies = [c.op]
    v01.var.dependents = [c.op]
    c.op.dependencies = [v01.var]
    c.op.dependents = [v23.var]
    c.op.outs = (v23.var, )

    graph.remove_edges_from([
        (v01, op1),
        (op1, v12),
        (v12, op2),
        (op2, v23),
    ])
    graph.remove_nodes_from([op1, v12, op2])
    graph.add_edges_from([
        (v01, c),
        (c, v23),
    ])


def define_compute_strings(graph: DiGraph):
    ops = get_operation_nodes(graph)
    for op in ops:
        if isinstance(op.op, StandardOperation):
            try:
                op.op.define_compute_strings()
            except NotImplementedError:
                pass


def combine_operations(rep: GraphRepresentation) -> GraphRepresentation:
    # flat
    # KLUDGE: make deep copy because we are still using the front end
    # graph representation in csdl_om
    graph = deepcopy(rep.flat_graph)
    registered_outputs = set(get_registered_outputs_from_graph(graph))
    define_compute_strings(graph)

    for r in registered_outputs:
        _combine_operations(
            graph,
            registered_outputs,
            r,
        )

    # unflat
    # KLUDGE: make deep copy because we are still using the front end
    # graph representation in csdl_om
    graph = deepcopy(rep.unflat_graph)
    registered_outputs = set(get_registered_outputs_from_graph(graph))
    combine_operations_hierarchical(
        graph,
        registered_outputs,
    )

    # implicit
    apply_fn_to_implicit_operation_nodes(rep, combine_operations)

    return rep


def combine_operations_hierarchical(
    graph: DiGraph,
    registered_outputs: set[VariableNode],
):
    model_nodes = get_model_nodes_from_graph(graph)
    for m in model_nodes:
        rr = set(get_registered_outputs_from_graph(m.graph))
        combine_operations_hierarchical(m.graph, rr)
    for r in registered_outputs:
        define_compute_strings(graph)
        _combine_operations(graph, registered_outputs, r)


def _combine_operations(
    graph: DiGraph,
    registered_outputs: Set[VariableNode],
    v23: VariableNode,
):
    if isinstance(v23.var, Output):
        op2: OperationNode = list(graph.predecessors(v23))[0]
        if combinable(graph, registered_outputs, op2):
            v12: VariableNode = list(graph.predecessors(op2))[0]
            if isinstance(v12.var, Output):
                op1: OperationNode = list(graph.predecessors(v12))[0]
                if combinable(graph, registered_outputs, op1):
                    v01: VariableNode = list(graph.predecessors(op1))[0]
                    insert_new_op(
                        graph,
                        v01,
                        op1,
                        v12,
                        op2,
                        v23,
                    )
                    _combine_operations(
                        graph,
                        registered_outputs,
                        v01,
                    )
                else:
                    # make a copy of predecessors so that object over
                    # which loop iterates does not change size on each
                    # iteration
                    p = list(graph.predecessors(op1))
                    for v01 in p:
                        _combine_operations(
                            graph,
                            registered_outputs,
                            v01,
                        )
        else:
            # make a copy of predecessors so that object over which loop
            # iterates does not change size on each iteration
            p = graph.predecessors(op2)
            for v12 in p:
                _combine_operations(
                    graph,
                    registered_outputs,
                    v12,
                )

    #             # registered outputs cannot be eliminated when combining
    #             # operations
    #             if v12 in registered_outputs:
    #                 skip = True
    #             else:
    #                 # a variable will only have a preceding operation if
    #                 # it is an output
    #                 if isinstance(v12, VariableNode):
    #                     if isinstance(v12.var, Output):
    #                         # each variable only has one predecessor, which
    #                         # is always an operation
    #                         op1: OperationNode = list(
    #                             graph.predecessors(v12))[0]
    #                         vars1: list[VariableNode] = list(
    #                             graph.predecessors(v23))
    #                         # an operation must meet criteria for being
    #                         # eligible to be combined with a subsequent
    #                         # operation
    #                         if combinable(op1.op, vars1):
    #                             v01: VariableNode = vars2[0]
    #                             # this is where the action is
    #                             insert_new_op(
    #                                 graph,
    #                                 v01,
    #                                 op1,
    #                                 v12,
    #                                 op2,
    #                                 v23,
    #                             )
    #                         else:
    #                             # if an operation cannot be combined, find
    #                             # the two preceding operations for each
    #                             # argument to this operation that can be
    #                             # combined
    #                             for dep in vars2:
    #                                 combine_operations_this_level(
    #                                     graph, registered_outputs, dep)
    #                     else:
    #                         # if variable is not an output, skip it
    #                         skip = True
    #                 else:
    #                     # if node is  not a variable, skip it
    #                     skip = True
    #         else:
    #             # if an operation cannot be combined, find the two
    #             # preceding operations for each argument to this
    #             # operation that can be combined
    #             for dep in vars2:
    #                 combine_operations_this_level(
    #                     graph,
    #                     registered_outputs,
    #                     dep,
    #                 )
    #     else:
    #         # if variable is not an output, skip it
    #         skip = True
    # else:
    #     # if node is not a variable node, skip it
    #     skip = True
    # # TODO: when and how to skip?
    # # if skip is True:
    # for dep in graph.predecessors(v23):
    #     combine_operations_this_level(graph, registered_outputs, dep)
