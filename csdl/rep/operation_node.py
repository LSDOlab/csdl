from csdl.rep.ir_node import IRNode
from csdl.lang.operation import Operation


class OperationNode(IRNode):

    def __init__(self, op: Operation):
        super().__init__()
        self.op: Operation = op
        self.name = op.name
