from csdl.core.variable import Variable
from typing import TypeVar

V = TypeVar('V', bound=Variable)


class VariableNode:

    def __init__(self, var: V):
        self.var: V = var
