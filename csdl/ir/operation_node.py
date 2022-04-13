from lib2to3.pgen2.token import OP
from csdl.ir.node import Node
from csdl.core.operation import Operation


class OperationNode(Node):

    def __init__(self, op: Operation):
        super().__init__()
        self.op: Operation = op
