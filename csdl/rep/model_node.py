from csdl.rep.ir_node import IRNode
try:
    from csdl.lang.model import Model
except ImportError:
    pass
from networkx import DiGraph
from typing import Tuple, Dict, Any


class ModelNode(IRNode):

    def __init__(self, name: str, model: 'Model',
                 promotes: list[str] | None):
        super().__init__()
        self.name: str = name
        self.model: Model = model
        self.connections: list[Tuple[str, str]] = model.connections
        self.design_variables: Dict[str,
                                    Dict[str,
                                         Any]] = model.design_variables
        self.constraints: Dict[str, Dict[str, Any]] = model.constraints
        self.objective: Dict[str, Any] = model.objective
        self.graph: DiGraph = DiGraph()
        self.sorted_nodes: list[IRNode] = []
        self.promotes: list[str] | None = promotes
