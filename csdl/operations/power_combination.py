from csdl.core.standard_operation import StandardOperation
from numbers import Number
import numpy as np
from csdl.core.node import Node
from csdl.core.output import Output
from csdl.utils.gen_hex_name import gen_hex_name


class power_combination(StandardOperation):
    def __init__(self, *args, powers, coeff, **kwargs):
        self.nouts = 1
        self.nargs = None
        super().__init__(*args, **kwargs)
        dep0 = self.dependencies[0]
        for dep in self.dependencies:
            if dep0.shape != dep.shape:
                raise ValueError(
                    "Shapes of inputs to linear_combination do not match")
        self.properties['elementwise'] = True

        self.outs = [
            Output(
                None,
                op=self,
                shape=self.dependencies[0].shape,
            )
        ]

        self.literals['powers'] = powers
        self.literals['coeff'] = coeff

    def define_compute_strings(self):
        out_name = self.outs[0].name
        self.compute_string = '{}='.format(out_name)
        args = self.dependencies
        powers = self.literals['powers']
        coeff = self.literals['coeff']
        # if isinstance(constant, np.ndarray):
        #     raise notimplementederror("constant must be a scalar constant")
        if isinstance(powers, (int, float)):
            powers = [powers] * len(args)
        self.compute_string = '{}={}'.format(out_name, coeff)
        for arg, power in zip(args, powers):
            if not np.all(coeff == 0):
                self.compute_string += '*{}**{}'.format(arg.name, power)
            else:
                self.compute_string = '0'
