from csdl.core.standard_operation import StandardOperation


class reorder_axes(StandardOperation):
    def __init__(self, *args, operation, new_axes_locations, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
                self.literals['operation'] = operation
        self.literals['new_axes_locations'] = new_axes_locations
