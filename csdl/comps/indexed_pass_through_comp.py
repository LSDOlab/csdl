import numpy as np
from openmdao.api import ExplicitComponent

from csdl.core.variable import Variable


class IndexedPassThroughComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('out_name', types=str)
        self.options.declare('out_shape', types=tuple)
        self.options.declare('expr_indices', types=dict)
        self.options.declare('vals', types=dict)

    def setup(self):
        out_name = self.options['out_name']
        out_shape = self.options['out_shape']
        expr_indices = self.options['expr_indices']
        vals = self.options['vals']
        self.add_output(
            out_name,
            shape=out_shape,
            # units=out_expr.units,
        )

        # TODO: How exactly does Python interpret this? Is it done
        # lazily, or is there a real performance impact?
        z = zip(expr_indices.items(), vals.items())
        for (in_name, (shape, tgt_indices)), (_, val) in list(z):
            self.add_input(
                in_name,
                shape=shape,
                val=val,
                # units=expr.units,
            )
            self.declare_partials(
                out_name,
                in_name,
                val=1.,
                rows=tgt_indices,
                cols=np.arange(len(tgt_indices)),
            )

    def compute(self, inputs, outputs):
        out_name = self.options['out_name']
        out_shape = self.options['out_shape']
        expr_indices = self.options['expr_indices']
        for in_name, (shape, tgt_indices) in expr_indices.items():
            i = np.unravel_index(tgt_indices, out_shape)
            outputs[out_name][i] = inputs[in_name].flatten()
