from csdl.core.operation import Operation
from csdl.core.output import Output

from typing import List, Tuple


class StandardOperation(Operation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outs: List[Output] = None
        self.literals = dict()
