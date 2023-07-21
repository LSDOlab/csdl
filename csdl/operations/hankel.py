from csdl.lang.standard_operation import StandardOperation

class hankel(StandardOperation):

    def __init__(self, *args, kind, order, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.properties['elementwise'] =  True
        self.literals['kind'] = kind
        self.literals['order'] = order

'''
NOTE:
- 3 inputs:
    - hankel function type (integer, 1 or 2) ----> DOES NOT CHANGE WITH OPTIMIZATION ITERATION
    - order (integer) ----> DOES NOT CHANGE WITH OPTIMIZATION ITERATION
    - val (float) ----> CSDL VARIABLE, will change
'''