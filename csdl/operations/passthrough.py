from csdl.lang.standard_operation import StandardOperation
from csdl.lang.node import Node


class passthrough(StandardOperation):

    def __init__(self, *args, **kwargs):
        name = 'passthrough'
        self.nargs = 1
        self.nouts = 1
        super().__init__(*args, **kwargs)
        if len(args) != self.nargs:
            raise ValueError("{} takes exactly {} arguments".format(
                name, self.nargs))
        self.iterative = True
        self.properties['elementwise'] = True
