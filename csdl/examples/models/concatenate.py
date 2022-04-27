from csdl import Model
import numpy as np


class ConcatenateFunction(Model):

    def define(self):

        b = self.declare_variable('b')
        e = self.declare_variable('e')

        c = self.create_output('c', shape=(2,))
        c[0] = b + e
        c[1] = e
