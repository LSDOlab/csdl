from csdl.lang.operation import Operation
from csdl.lang.output import Output

from typing import List, Tuple


class StandardOperation(Operation):

    def __init__(self, *args, **kwargs):

        # If there are repeating args, create a copy of already existing variables by multiplying them by one
        # This is necessary for operations like x*x, which have the same variable as an argument twice
        # The arguments should go from [x,x] to [x,1*x]
        unique_args = set()
        new_args = []
        for arg in args:
            if arg in unique_args:
                new_arg = arg * 1
                new_args.append(new_arg)
                unique_args.add(new_arg)
            else:
                unique_args.add(arg)
                new_args.append(arg)

        # Pass in variables as normal
        super().__init__(*new_args, **kwargs)
        self.literals = dict()
        self.compute_string = ''

    def define_compute_strings(self):
        raise NotImplementedError(
            "Compute strings are not defined for operation {}".format(
                type(self)))
