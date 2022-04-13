from csdl.ir.node import Node
from csdl.core.variable import Variable


class VariableNode(Node):

    def __init__(self, var: Variable):
        super().__init__()
        self.var: Variable = var
