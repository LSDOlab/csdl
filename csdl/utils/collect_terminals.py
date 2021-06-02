from typing import List

from csdl.core.node import Node
from csdl.core.variable import Variable
from csdl.core.operation import Operation


def isterminal(node: Node):
    # dependency is a variable that does not depend on subsystem
    return isinstance(node, Variable) and len(node.dependencies) == 0


def collect_terminals2(
    terminals: List[Variable],
    residual: Variable,
    op: Operation,
) -> List[Variable]:
    """
    Collect input nodes so that the resulting ``ImplicitComponent`` has
    access to inputs outside of itself.
    """
    for var in op.dependencies:
        # only collect terminals that are dependencies of the residual
        # and not the residual itself
        if var.name != residual.name:
            if isterminal(var):
                terminals.append(var)
            terminals = collect_terminals(terminals, residual, var)
    return terminals


def collect_terminals(
    terminals: List[Variable],
    residual: Variable,
    var: Variable,
) -> List[Variable]:
    """
    Collect input nodes so that the resulting ``ImplicitComponent`` has
    access to inputs outside of itself.
    """
    for op in var.dependencies:
        terminals = collect_terminals2(terminals, residual, op)
    return terminals
