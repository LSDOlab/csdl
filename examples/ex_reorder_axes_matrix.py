from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class ExampleMatrix(Model):
    def define(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_variable(
            'M1',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Compute the transpose of mat
        self.register_output('axes_reordered_matrix',
                             csdl.reorder_axes(mat, 'ij->ji'))


sim = Simulator(ExampleMatrix())
sim.run()

print('M1', sim['M1'].shape)
print(sim['M1'])
print('axes_reordered_matrix', sim['axes_reordered_matrix'].shape)
print(sim['axes_reordered_matrix'])
