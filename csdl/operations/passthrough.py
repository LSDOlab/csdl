from csdl.core.standard_operation import StandardOperation
from csdl.core.node import Node


class passthrough(StandardOperation):
    def __init__(self, *args, output, **kwargs):
        name = 'passthrough'
        self.nargs = 1
        self.nouts = 1
        super().__init__(*args, **kwargs)
        if len(args) != self.nargs:
            raise ValueError("{} takes exactly {} arguments".format(
                name, self.nargs))
        self.outs = [output]
