import itertools
from networkx import DiGraph
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.implicit_operation_node import ImplicitOperationNode
from csdl.lang.bracketed_search_operation import BracketedSearchOperation


def create_graph_with_operations_as_edges(g: DiGraph):
    ops = [x for x in g.nodes if isinstance(x, OperationNode)]
    vars = [x for x in g.nodes if isinstance(x, VariableNode)]

    new_g = DiGraph()
    new_g.add_nodes_from(vars)

    for op in ops:
        a = list(g.predecessors(op))
        b = list(g.successors(op))
        cost = 1
        if isinstance(op, ImplicitOperationNode):
            maxiter = op.op.maxiter if isinstance(
                op.op, BracketedSearchOperation
            ) else op.op.nonlinear_solver.options['maxiter']
            cost = op.rep.critical_path_length() * maxiter
        new_g.add_edges_from(list(itertools.product(a, b)), weight=cost)

    return new_g
