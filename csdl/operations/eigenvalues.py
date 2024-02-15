from csdl.lang.standard_operation import StandardOperation


class eigenvalues(StandardOperation):

    def __init__(self, *args, **kwargs):
        self.nouts = 2
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.properties['elementwise'] = False
        self.literals['n'] = kwargs['n']


