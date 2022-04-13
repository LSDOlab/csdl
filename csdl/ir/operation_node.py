from csdl.core.operation import Operation
from typing import TypeVar

O = TypeVar('O', bound=Operation)


class OperationNode:

    def __init__(self, op: O):
        self.op: O = op
