from csdl.lang.standard_operation import StandardOperation


class average(StandardOperation):

    def __init__(self, *args, axes, **kwargs):
        self.nouts = 1
        self.nargs = None
        super().__init__(*args, **kwargs)
        self.literals['axes'] = axes
