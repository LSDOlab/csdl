from csdl.core.standard_operation import StandardOperation
from numbers import Number
import numpy as np
from csdl.core.output import Output
from csdl.utils.gen_hex_name import gen_hex_name


class linear_combination(StandardOperation):
    def __init__(self, *args, **kwargs):
        self.nouts = 1
        self.nargs = None
        super().__init__(*args, **kwargs)
        dep0 = self.dependencies[0]
        for dep in self.dependencies:
            if dep0.shape != dep.shape:
                raise ValueError(
                    "Shapes of inputs to linear_combination do not match")

        self.outs = [
            Output(
                None,
                op=self,
                shape=self.dependencies[0].shape,
            )
        ]

        for k, v in kwargs.items():
            if k == 'constant':
                self.literals[k] = v
            if k == 'coeffs':
                self.literals[k] = v
