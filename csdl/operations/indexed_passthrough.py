from csdl.core.standard_operation import StandardOperation
from csdl.core.node import Node


class indexed_passthrough(StandardOperation):
    def __init__(self, *args, output, **kwargs):
        self.nargs = None
        self.nouts = 1
        super().__init__(*args, **kwargs)
        self.outs = [output]
        self.properties['iterative'] = False
