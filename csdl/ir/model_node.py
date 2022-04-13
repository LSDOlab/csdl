from csdl.ir.node import Node
from csdl.core.model import Model
from networkx import DiGraph


class ModelNode(Node):

    def __init__(self, model: Model):
        super().__init__()
        self.model: Model = model
        self.graph: DiGraph | None = None
