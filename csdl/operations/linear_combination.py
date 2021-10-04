from csdl.core.standard_operation import StandardOperation
from numbers import Number
import numpy as np
from csdl.core.output import Output
from csdl.utils.gen_hex_name import gen_hex_name


class linear_combination(StandardOperation):
    def __init__(self, *args, constant, coeffs, **kwargs):
        self.nouts = 1
        self.nargs = None
        super().__init__(*args, **kwargs)
        dep0 = self.dependencies[0]
        for dep in self.dependencies:
            if dep0.shape != dep.shape:
                raise ValueError(
                    "Shapes of inputs to linear_combination do not match"
                )
        self.properties['iterative'] = False
        self.properties['elementwise'] = True

        self.literals['constant'] = constant
        self.literals['coeffs'] = coeffs

    def define_compute_strings(self):
        out_name = self.outs[0].name
        self.compute_string = '{}='.format(out_name)
        args = self.dependencies
        coeffs = self.literals['coeffs']
        constant = self.literals['constant']
        # if isinstance(constant, np.ndarray):
        #     raise notimplementederror("constant must be a scalar constant")
        if isinstance(coeffs, (int, float)):
            coeffs = [coeffs] * len(args)
        if np.all(constant == 0):
            self.compute_string = '{}='.format(out_name, constant)
        else:
            self.compute_string = '{}={}'.format(out_name, constant)
        first = True
        for coeff, arg in zip(coeffs, args):
            if isinstance(coeff, (int, float)):
                if coeff > 1:
                    if first and np.all(constant == 0):
                        self.compute_string += '{}*{}'.format(
                            coeff, arg.name)
                        first = False
                    else:
                        self.compute_string += '+{}*{}'.format(
                            coeff, arg.name)
                elif coeff > 0:
                    if first and np.all(constant == 0):
                        self.compute_string += '{}'.format(arg.name)
                        first = False
                    else:
                        self.compute_string += '+{}'.format(arg.name)
                elif coeff < -1:
                    self.compute_string += '{}*{}'.format(
                        coeff, arg.name)
                elif coeff < 0:
                    self.compute_string += '-{}'.format(arg.name)
            else:
                if not np.all(coeff == 0):
                    if first and np.all(constant == 0):
                        self.compute_string += '{}*{}'.format(
                            coeff, arg.name)
                        first = False
                    else:
                        self.compute_string += '+{}*{}'.format(
                            coeff, arg.name)
