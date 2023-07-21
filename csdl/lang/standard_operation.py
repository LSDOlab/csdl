from csdl.lang.operation import Operation
from csdl.lang.output import Output

from typing import List, Tuple


class StandardOperation(Operation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.literals = dict()
        self.compute_string = ''

    def define_compute_strings(self):
        raise NotImplementedError(
            "Compute strings are not defined for operation {}".format(
                type(self)))
