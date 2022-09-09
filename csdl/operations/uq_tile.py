from csdl.lang.standard_operation import StandardOperation


class uq_tile(StandardOperation):

    def __init__(self, *args, einsum_string=None, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.literals['einsum_string'] = einsum_string