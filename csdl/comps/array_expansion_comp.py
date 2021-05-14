import numpy as np

from openmdao.api import ExplicitComponent

from csdl.utils.get_array_indices import get_array_indices
from csdl.utils.decompose_shape_tuple import decompose_shape_tuple


class ArrayExpansionComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('expand_indices', types=list)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        shape = self.options['shape']
        expand_indices = self.options['expand_indices']
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        val = self.options['val']

        (
            in_string,
            ones_string,
            out_string,
            in_shape,
            ones_shape,
            out_shape,
        ) = decompose_shape_tuple(shape, expand_indices)

        einsum_string = '{},{}->{}'.format(in_string, ones_string, out_string)

        self.add_input(in_name, shape=in_shape, val=val)
        self.add_output(out_name, shape=out_shape)

        in_indices = get_array_indices(*in_shape)
        out_indices = get_array_indices(*out_shape)

        self.einsum_string = einsum_string
        self.ones_shape = ones_shape

        rows = out_indices.flatten()
        cols = np.einsum(einsum_string, in_indices, np.ones(ones_shape,
                                                            int)).flatten()
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.einsum(self.einsum_string, inputs[in_name],
                                      np.ones(self.ones_shape))
