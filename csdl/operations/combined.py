from csdl.core.standard_operation import StandardOperation


class combined(StandardOperation):
    def __init__(self, *args, **kwargs):
        self.nouts = 1
        self.nargs = None
        super().__init__(*args, **kwargs)

    def define_compute_strings(self):
        # defined by csdl.utils.combine_operations
        pass
