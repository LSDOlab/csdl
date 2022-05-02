from csdl.rep.ir_node import IRNode
try:
    from csdl.lang.model import Model
except ImportError:
    pass
from networkx import DiGraph
from typing import Tuple


class ModelNode(IRNode):

    def __init__(self, name: str, model: 'Model'):
        super().__init__()
        self.name: str = name
        self.model: Model = model
        self.connections: list[Tuple[str, str]] = model.connections
        self.design_variables = model.design_variables
        self.constraints = model.constraints
        self.objective = model.objective
        self.graph: DiGraph = DiGraph()
        self.sorted_nodes: list[IRNode]
