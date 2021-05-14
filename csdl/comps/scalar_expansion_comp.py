import numpy as np

from openmdao.api import ExplicitComponent

from csdl.utils.get_array_indices import get_array_indices


class ScalarExpansionComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        shape = self.options['shape']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        self.add_input(in_name)
        self.add_output(out_name, shape=shape)

        rows = np.arange(np.prod(shape))
        cols = np.zeros(np.prod(shape), int)
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = inputs[in_name]
