from csdl.core.operation import Operation
from csdl.core.output import Output

from typing import List, Tuple


class StandardOperation(Operation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outs: List[Output] = []
        self.literals = dict()
        self.compute_string = ''
        self.compute_derivs = dict()
        self.properties = set()
        self.step = 1e-40

    def define_compute_strings(self):
        raise NotImplementedError(
            "Compute strings are not defined for operation {}".format(
                type(self)))
