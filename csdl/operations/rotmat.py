from csdl.lang.standard_operation import StandardOperation


class rotmat(StandardOperation):

    def __init__(self, *args, axis, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.literals['axis'] = axis
