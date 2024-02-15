from csdl.lang.standard_operation import StandardOperation
from csdl.lang.variable import Variable

class exp_a(StandardOperation):

    def __init__(self, var, a=None, **kwargs):
        self.nouts = 1
        self.nargs = 1

        args = [var]
        if isinstance(a, Variable):
            args.append(a)
            self.nargs = 2

        super().__init__(*args, **kwargs)
        self.properties['elementwise'] = True
        self.literals['a'] = a

    def define_compute_strings(self):
        in_name = self.dependencies[0].name
        out_name = self.outs[0].name
        literals = self.literals
        self.compute_string = '{}=({}**{})'.format(out_name, literals['a'], in_name) 