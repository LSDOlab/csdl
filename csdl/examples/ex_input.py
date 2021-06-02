from csdl import Model
import numpy as np


class ExampleSimple(Model):
    """
    :param var: z
    """
    def define(self):
        z = self.create_input('z', val=10)
