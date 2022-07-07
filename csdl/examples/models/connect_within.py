from csdl import Model
import numpy as np


class ConnectWithin(Model):

    def define(self):

        # inputs
        a = self.declare_variable('a')
        b = self.create_input('b')

        f = a + b

        # outputs
        c = self.declare_variable('c')
        self.register_output('y', f+c)
        self.connect('b', 'c')
