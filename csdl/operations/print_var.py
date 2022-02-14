from csdl.core.standard_operation import StandardOperation
from csdl.core.node import Node


class print_var(StandardOperation):

    def __init__(self, *args, **kwargs):
        name = 'print_val'
        self.nargs = 1
        self.nouts = 1
        super().__init__(*args, **kwargs)
        if len(args) != self.nargs:
            raise ValueError("{} takes exactly {} arguments".format(
                name, self.nargs))
        self.properties['elementwise'] = True

    def define_compute_strings(self):
        pass
