from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class ExampleTensor(Model):
    def define(self):

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_variable(
            'T1',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute a new tensor by reordering axes of tens; reordering is
        # given by 'ijkl->ljki'
        self.register_output('axes_reordered_tensor',
                             csdl.reorder_axes(tens, 'ijkl->ljki'))


sim = Simulator(ExampleTensor())
sim.run()

print('T1', sim['T1'].shape)
print(sim['T1'])
print('axes_reordered_tensor', sim['axes_reordered_tensor'].shape)
print(sim['axes_reordered_tensor'])
