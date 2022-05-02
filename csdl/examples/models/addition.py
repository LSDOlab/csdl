from csdl import Model, GraphRepresentation
import numpy as np


class AdditionFunction(Model):

    def define(self):

        # inputs
        a = self.declare_variable('a')
        b = self.declare_variable('b')

        f = a + b

        # outputs
        self.register_output('f', f)


class AdditionVectorFunction(Model):

    def define(self):

        # inputs
        a = self.declare_variable('a', shape=(3, ))
        b = self.declare_variable('b', shape=(3, ))

        f = a + b

        # outputs
        self.register_output('f', f)
