import numpy as np
from openmdao.api import ExplicitComponent

from csdl.core.variable import Variable


class DecomposeComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('expr', types=Variable)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        expr = self.options['expr']
        in_name = self.options['in_name']
        val = self.options['val']
        self.add_input(
            in_name,
            shape=expr.shape,
            val=val,
            # units=in_expr.units,
        )

        for out_expr, src_indices in expr.src_indices.items():
            name = out_expr.name
            shape = out_expr.shape

            self.add_output(
                name,
                shape=shape,
                # units=expr.units,
            )
            self.declare_partials(
                name,
                in_name,
                val=1.,
                rows=np.arange(len(src_indices)),
                cols=src_indices,
            )

    def compute(self, inputs, outputs):
        expr = self.options['expr']
        in_name = self.options['in_name']
        for out_expr, src_indices in expr.src_indices.items():
            name = out_expr.name
            outputs[name] = inputs[in_name].flatten()[src_indices]
