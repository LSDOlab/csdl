from csdl.core.standard_operation import StandardOperation


class log10(StandardOperation):
    def __init__(self, *args, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.properties['iterative'] = False
        self.properties['elementwise'] = True
