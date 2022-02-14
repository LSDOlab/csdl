from csdl.core.standard_operation import StandardOperation


class inner(StandardOperation):
    def __init__(self, *args, axes, **kwargs):
        self.nouts = 1
        self.nargs = 2
        super().__init__(*args, **kwargs)
                self.literals['axes'] = axes
