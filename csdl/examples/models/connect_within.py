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

        # setting a connection within a model is not allowed
        self.connect('b', 'c')
