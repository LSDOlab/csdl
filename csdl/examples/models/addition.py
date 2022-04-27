from csdl import Model
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
        a = self.declare_variable('a', shape=(3,))
        b = self.declare_variable('b', shape=(3,))

        f = a + b

        # outputs
        self.register_output('f', f)


class ParallelAdditionFunction(Model):

    def define(self):

        # inputs
        y = self.declare_variable('y_in')
        x = self.declare_variable('x_in')

        # outputs
        self.register_output('y_out', y+1)
        self.register_output('x_out', x+1)
