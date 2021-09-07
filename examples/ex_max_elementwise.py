from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class ExampleElementwise(Model):
    def define(self):

        m = 2
        n = 3
        # Shape of the three tensors is (2,3)
        shape = (m, n)

        # Creating the values for two tensors
        val1 = np.array([[1, 5, -8], [10, -3, -5]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_variable('tensor1', val=val1)
        tensor2 = self.declare_variable('tensor2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMin', csdl.max(tensor1, tensor2))


sim = Simulator(ExampleElementwise())
sim.run()

print('tensor1', sim['tensor1'].shape)
print(sim['tensor1'])
print('tensor2', sim['tensor2'].shape)
print(sim['tensor2'])
print('ElementwiseMin', sim['ElementwiseMin'].shape)
print(sim['ElementwiseMin'])
