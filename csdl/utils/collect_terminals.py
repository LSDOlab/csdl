from typing import List

from csdl.lang.node import Node
from csdl.lang.variable import Variable
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.output import Output
from csdl.lang.operation import Operation


def isterminal(node: Node):
    # dependency is a variable that does not depend on subsystem
    return isinstance(node, Variable) and len(node.dependencies) == 0


def collect_terminals2(
    terminals: List[DeclaredVariable],
    residual: Output,
    op: Operation,
) -> List[DeclaredVariable]:
    """
    Collect input nodes so that the resulting ``ImplicitModel`` has
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
    terminals: List[DeclaredVariable],
    residual: Output,
    var: Variable,
) -> List[DeclaredVariable]:
    """
    Collect input nodes so that the resulting ``ImplicitModel`` has
    access to inputs outside of itself.
    """
    for op in var.dependencies:
        terminals = collect_terminals2(terminals, residual, op)
    return terminals
