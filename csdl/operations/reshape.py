from csdl.lang.standard_operation import StandardOperation


class reshape(StandardOperation):

    def __init__(self, *args, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
