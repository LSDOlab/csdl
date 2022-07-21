from csdl.lang.standard_operation import StandardOperation


class exp_a(StandardOperation):

    def __init__(self, *args, a=None, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.properties['elementwise'] = True
        self.literals['a'] = a

    def define_compute_strings(self):
        in_name = self.dependencies[0].name
        out_name = self.outs[0].name
        literals = self.literals
        self.compute_string = '{}=({}**{})'.format(out_name, literals['a'], in_name) 