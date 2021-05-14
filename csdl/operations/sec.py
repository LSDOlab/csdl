from csdl.core.standard_operation import StandardOperation


class sec(StandardOperation):
    def __init__(self, *args, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
