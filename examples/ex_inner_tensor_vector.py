from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class ExampleTensorVector(Model):
    def define(self):

        m = 3
        n = 4
        p = 5

        # Shape of the vectors
        vec_shape = (m, )

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Values for the two vectors
        vec1 = np.arange(m)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)

        # Adding the vector and tensor to csdl
        vec1 = self.declare_variable('vec1', val=vec1)
        ten1 = self.declare_variable('ten1', val=ten1)

        # Tensor-Vector Inner Product specifying the first axis for
        # Vector and Tensor
        self.register_output(
            'TenVecInner',
            csdl.inner(ten1, vec1, axes=([0], [0])),
        )


sim = Simulator(ExampleTensorVector())
sim.run()

print('vec1', sim['vec1'].shape)
print(sim['vec1'])
print('ten1', sim['ten1'].shape)
print(sim['ten1'])
print('TenVecInner', sim['TenVecInner'].shape)
print(sim['TenVecInner'])
