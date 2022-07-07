from csdl import Model, GraphRepresentation
import numpy as np


class FalseCyclePost(Model):

    def define(self):

        # inputs
        f = self.declare_variable('f')
        c = self.declare_variable('c')

        # outputs
        self.register_output('y', f + 1)
        self.register_output('x', c + 1)
