from csdl import Model
import csdl
import numpy as np


class ExampleSimple(Model):
    """
    :param var: vec1
    :param var: vec2
    :param var: VecVecCross
    """
    def define(self):
        x = self.declare_variable('x')
        y = self.declare_variable('y')

        a = x + y
        b = x + y
        c = 2 * a
        d = 2 * b

        self.register_output('c', c)
        self.register_output('d', d)
