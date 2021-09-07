from csdl_om import Simulator
import numpy as np
from csdl import Model
import csdl


class ExampleReorderTensor(Model):
    def define(self):

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        # Transpose of a tensor
        self.register_output('einsum_reorder2',
                             csdl.einsum(tens, subscripts='ijk->kji'))


sim = Simulator(ExampleReorderTensor())
sim.run()

print('c', sim['c'].shape)
print(sim['c'])
print('einsum_reorder2', sim['einsum_reorder2'].shape)
print(sim['einsum_reorder2'])
