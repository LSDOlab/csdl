from csdl.core.standard_operation import StandardOperation


class einsum(StandardOperation):
    def __init__(self, *args, subscripts, partial_format, **kwargs):
        self.nouts = 1
        self.nargs = None
        super().__init__(*args, **kwargs)
        self.properties['iterative'] = False
        self.literals['subscripts'] = subscripts
        self.literals['partial_format'] = partial_format
