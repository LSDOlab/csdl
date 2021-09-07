from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class ExampleMultipleMatrix(Model):
    def define(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_variable('M1', val=np.arange(n * m).reshape((n, m)))

        # Declare another matrix of shape 3x6 as input
        M2 = self.declare_variable('M2',
                                   val=np.arange(n * m, 2 * n * m).reshape(
                                       (n, m)))

        # Output the elementwise average of matrices M1 and M2
        self.register_output('multiple_matrix_average', csdl.average(M1, M2))


sim = Simulator(ExampleMultipleMatrix())
sim.run()

print('M1', sim['M1'].shape)
print(sim['M1'])
print('M2', sim['M2'].shape)
print(sim['M2'])
print('multiple_matrix_average', sim['multiple_matrix_average'].shape)
print(sim['multiple_matrix_average'])
