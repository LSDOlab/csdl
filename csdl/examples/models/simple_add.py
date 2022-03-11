from csdl import Model
import numpy as np


class SimpleAdd(Model):
    def initialize(self):
        self.parameters.declare(
            'p',
            types=(int, float, np.ndarray, list),
        )
        self.parameters.declare(
            'q',
            types=(int, float, np.ndarray, list),
        )

    def define(self):
        p = self.create_input('p', val=self.parameters['p'])
        q = self.create_input('q', val=self.parameters['q'])
        r = p + q
        self.register_output('r', r)
