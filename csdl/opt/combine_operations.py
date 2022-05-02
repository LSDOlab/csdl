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
from csdl.rep.get_model_nodes_from_graph import get_model_nodes_from_graph
from csdl.rep.graph_representation import GraphRepresentation
from csdl.rep.apply_fn_to_implicit_operation_nodes import apply_fn_to_implicit_operation_nodes


def combinable(op: Operation, vars: list[VariableNode]):
    if len(vars) != 1:
        return False
    if isinstance(op, StandardOperation):
        return check_property(op, 'elementwise', True)
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
        op1.op.define_compute_strings()
        new_op.compute_string += op1.op.compute_string + '\n'
    if isinstance(op2.op, StandardOperation):
        op2.op.define_compute_strings()
        new_op.compute_string += op2.op.compute_string + '\n'
    if isinstance(v23.var, Output):
        new_op.outs = (v23.var, )
        new_op.dependents = [v23.var]
    new_op_node = OperationNode(new_op)

    # update dependencies encoded in linked list implementation of graph
    # representation
    v23.var.dependencies = []
    v23.var.add_dependency_node(new_op)
    new_op.add_dependency_node(v01.var)
    v01.var.dependents = [new_op]

    # update edges in networkx implementation of graph representation
    graph.add_edge(v01, new_op_node)
    graph.add_edge(new_op_node, v23)

    # make sure reference counts for removed operations and variables go
    # to zero when this function terminates
    op1.op.dependencies = []
    v12.var.dependencies = []
    op2.op.dependencies = []
    op1.op.dependents = []
    v12.var.dependents = []
    op2.op.dependents = []

    graph.remove_nodes_from([op1, v12, op2])


def combine_operations(rep: GraphRepresentation) -> GraphRepresentation:
    # flat
    registered_outputs = set(
        get_registered_outputs_from_graph(rep.flat_graph))
    for r in registered_outputs:
        combine_operations_this_level(
            rep.flat_graph,
            registered_outputs,
            r,
        )

    # unflat
    registered_outputs = set(
        get_registered_outputs_from_graph(rep.unflat_graph))
    for r in registered_outputs:
        combine_operations_hierarchical(
            rep.unflat_graph,
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
        combine_operations_this_level(graph, registered_outputs, r)


def combine_operations_this_level(
    graph: DiGraph,
    registered_outputs: Set[VariableNode],
    v23: IRNode,
):
    skip = False
    # given a variable, combine the two preceding operations
    if isinstance(v23, VariableNode):
        # a variable will only have a preceding operation if it is an
        # output
        if isinstance(v23.var, Output):
            # each variable only has one predecessor, which is always an
            # operation
            op2: OperationNode = list(graph.predecessors(v23))[0]
            vars2: list[VariableNode] = list(graph.predecessors(v23))
            # an operation must meet criteria for being eligible to be
            # combined with a previous operation
            if combinable(op2.op, vars2):
                v12: VariableNode = vars2[0]
                # registered outputs cannot be eliminated when combining
                # operations
                if v12 in registered_outputs:
                    skip = True
                else:
                    # a variable will only have a preceding operation if
                    # it is an output
                    if isinstance(v12, VariableNode):
                        if isinstance(v12.var, Output):
                            # each variable only has one predecessor, which
                            # is always an operation
                            op1: OperationNode = list(
                                graph.predecessors(v12))[0]
                            vars1: list[VariableNode] = list(
                                graph.predecessors(v23))
                            # an operation must meet criteria for being
                            # eligible to be combined with a subsequent
                            # operation
                            if combinable(op1.op, vars1):
                                v01: VariableNode = vars2[0]
                                # this is where the action is
                                insert_new_op(
                                    graph,
                                    v01,
                                    op1,
                                    v12,
                                    op2,
                                    v23,
                                )
                            else:
                                # if an operation cannot be combined, find
                                # the two preceding operations for each
                                # argument to this operation that can be
                                # combined
                                for dep in vars2:
                                    combine_operations_this_level(
                                        graph, registered_outputs, dep)
                        else:
                            # if variable is not an output, skip it
                            skip = True
                    else:
                        # if node is  not a variable, skip it
                        skip = True
            else:
                # if an operation cannot be combined, find the two
                # preceding operations for each argument to this
                # operation that can be combined
                for dep in vars2:
                    combine_operations_this_level(
                        graph,
                        registered_outputs,
                        dep,
                    )
        else:
            # if variable is not an output, skip it
            skip = True
    else:
        # if node is not a variable node, skip it
        skip = True
    # TODO: when and how to skip?
    # if skip is True:
    for dep in graph.predecessors(v23):
        combine_operations_this_level(graph, registered_outputs, dep)
