from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class ExampleMatrix(Model):
    def define(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_variable(
            'Mat',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Compute the transpose of mat
        self.register_output('matrix_transpose', csdl.transpose(mat))


sim = Simulator(ExampleMatrix())
sim.run()

print('Mat', sim['Mat'].shape)
print(sim['Mat'])
print('matrix_transpose', sim['matrix_transpose'].shape)
print(sim['matrix_transpose'])
