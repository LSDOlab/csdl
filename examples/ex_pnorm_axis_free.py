from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class ExampleAxisFree(Model):
    def define(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_variable('in1', val=tensor)

        # Computing the 6-norm on in1 without specifying an axis
        self.register_output('axis_free_pnorm', csdl.pnorm(in1, pnorm_type=6))


sim = Simulator(ExampleAxisFree())
sim.run()

print('in1', sim['in1'].shape)
print(sim['in1'])
print('axis_free_pnorm', sim['axis_free_pnorm'].shape)
print(sim['axis_free_pnorm'])
