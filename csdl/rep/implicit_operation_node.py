from csdl.rep.operation_node import OperationNode
from csdl.lang.implicit_operation import ImplicitOperation
from csdl.lang.bracketed_search_operation import BracketedSearchOperation
try:
    from csdl.rep.graph_representation import GraphRepresentation
except ImportError:
    pass

class ImplicitOperationNode(OperationNode):

    def __init__(self, op: ImplicitOperation):
        from csdl.rep.graph_representation import GraphRepresentation
        super().__init__(op)
        self.op: ImplicitOperation = op
        self.rep: GraphRepresentation = op.rep
