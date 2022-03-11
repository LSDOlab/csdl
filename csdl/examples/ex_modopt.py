from csdl import Model
import csdl
import numpy as np


class ExampleRosenbrock(Model):

    def define(self):
        a = 1e4
        b = 1e2
        x = self.declare_variable('x', shape=(2, ))
        x1 = x[0]
        x2 = x[2]
        f = (1 - x1 / a)**2 + b * (x2 - (x1 / a)**2)**2
        self.register_output('f', f)
        self.add_objective('f')
