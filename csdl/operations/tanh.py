from csdl.lang.standard_operation import StandardOperation


class tanh(StandardOperation):

    def __init__(self, *args, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.properties['elementwise'] = True

    def define_compute_strings(self):
        in_name = self.dependencies[0].name
        out_name = self.outs[0].name
        self.compute_string = '{}=np.tanh({})'.format(out_name, in_name)
