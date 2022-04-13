from csdl.core.standard_operation import StandardOperation


class pnorm(StandardOperation):

    def __init__(self, *args, pnorm_type, axis, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.literals['pnorm_type'] = pnorm_type
        self.literals['axis'] = axis
