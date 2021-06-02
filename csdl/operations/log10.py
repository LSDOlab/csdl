from csdl.core.standard_operation import StandardOperation


class log10(StandardOperation):
    def __init__(self, *args, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.properties['iterative'] = False
        self.properties['elementwise'] = True

    def define_compute_strings(self):
        in_name = self.dependencies[0].name
        out_name = self.outs[0].name
        self.compute_string = '{}=np.log10({})'.format(out_name, in_name)
