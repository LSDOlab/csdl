from csdl.core.standard_operation import StandardOperation


class outer(StandardOperation):
    def __init__(self, *args, **kwargs):
        self.nouts = 1
        self.nargs = 2
        super().__init__(*args, **kwargs)
        self.properties['iterative'] = False
