from csdl.rep.ir_node import IRNode
from csdl.lang.variable import Variable
from typing import Set

class VariableNode(IRNode):

    def __init__(self, var: Variable):
        super().__init__()
        self.var: Variable = var
        self.name = var.name
        self.unpromoted_namespace = ''
        self.tgt_namespace = []
        self.tgt_name = []

        self.connected_to: Set[VariableNode] = set()
