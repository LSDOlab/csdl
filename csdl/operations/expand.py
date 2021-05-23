from csdl.core.standard_operation import StandardOperation


class expand(StandardOperation):
    def __init__(self, *args, expand_indices=None, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.literals['expand_indices'] = expand_indices

    def define_compute_strings(self):
        pass
