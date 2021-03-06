from csdl.lang.standard_operation import StandardOperation


class max(StandardOperation):

    def __init__(self, *args, rho, axis, **kwargs):
        self.nouts = 1
        self.nargs = None
        super().__init__(*args, **kwargs)
        self.literals['axis'] = axis
        self.literals['rho'] = rho
