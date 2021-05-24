from csdl.core.standard_operation import StandardOperation


class cross(StandardOperation):
    def __init__(self, *args, axis, **kwargs):
        self.nouts = 1
        self.nargs = 2
        super().__init__(*args, **kwargs)
        self.properties['iterative'] = False
        self.literals['axis'] = axis
