from csdl.lang.standard_operation import StandardOperation
from csdl.lang.variable import Variable


class SolveLinear(StandardOperation):

    def __init__(self, A, b, n, solver):
        self.nouts = 1
        self.nargs = 0

        args = []
        csdl_ins = []
        if isinstance(A, Variable):
            args.append(A)
            self.nargs += 1
            csdl_ins.append('A')
        if isinstance(b, Variable):
            args.append(b)
            self.nargs += 1
            csdl_ins.append('b')

        super().__init__(*args)

        a_is_var = False
        a_name = None
        if isinstance(A, Variable):
            a_is_var = True
            a_name = A.name

        b_is_var = False
        b_name = None
        if isinstance(b, Variable):
            b_is_var = True
            b_name = b.name

        self.literals['A_info'] = {
            'obj': A,
            'var': a_is_var,
            'name': a_name,
        }
        self.literals['b_info'] = {
            'obj': b,
            'var': b_is_var,
            'name': b_name,
        }
        
        self.literals['csdl_ins'] = csdl_ins
        self.literals['n'] = n
        self.literals['solver'] = solver
