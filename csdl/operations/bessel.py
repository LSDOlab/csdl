from csdl.lang.standard_operation import StandardOperation

class bessel(StandardOperation):

    def __init__(self, *args, kind, order, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.properties['elementwise'] = True
        self.literals['kind'] = kind
        self.literals['order'] = order

    # def define_compute_strings(self):
    #     in_name_1 = self.dependencies[0].name # bessel function kind (integer, DOESN'T CHANGE)
    #     in_name_2 = self.dependencies[1].name # order (integer, DOESN'T CHANGE)
    #     in_name_3 = self.dependencies[2].name # value (this is a CSDL variable)
    #     out_name = self.outs[0].name
    #     # need if clause for the bessel function kind
    #     if in_name_1 == 1:
    #         self.compute_string = '{}=scipy.special.jv({},{})'.format(
    #             out_name, in_name_2, in_name_3)
    #     elif in_name_1 == 2:
    #         self.compute_string = '{}=scipy.special.yv({},{})'.format(
    #             out_name, in_name_2, in_name_3)
'''
NOTE:
- 3 inputs:
    - bessel function type (integer, 1 or 2) ----> DOES NOT CHANGE WITH OPTIMIZATION ITERATION
        - 1st type: J
        - 2nd type: Y
    - order (integer) ----> DOES NOT CHANGE WITH OPTIMIZATION ITERATION
    - val (float) ----> CSDL VARIABLE, will change
'''