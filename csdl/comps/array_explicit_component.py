import numpy as np

from openmdao.api import ExplicitComponent


class ArrayExplicitComponent(ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)

        self.array_initialize()

    def array_add_input(self, name, *args, **kwargs):
        self.add_input(name, *args, **kwargs, shape=self.var_shape)

    def array_add_output(self, name, *args, **kwargs):
        self.add_output(name, *args, **kwargs, shape=self.var_shape)

    def array_declare_partials(self, out_name, in_name, val=1.):
        arange = np.arange(self.var_size)
        self.declare_partials(out_name, in_name, val=val, rows=arange, cols=arange)

    def setup(self):
        self.var_shape = self.options['shape']
        self.var_size = np.prod(self.options['shape'])

        self.array_setup()

        self.set_check_partial_options('*', method='cs')

    def array_initialize(self):
        pass

    def array_setup(self):
        pass