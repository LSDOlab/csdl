from csdl_om import Simulator
import numpy as np
from csdl import Model
import csdl


class ExampleTensorSummationSparse(Model):
    def define(self):
        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        self.register_output(
            'einsum_summ2_sparse_derivs',
            csdl.einsum_new_api(tens,
                                operation=[(33, 66, 99)],
                                partial_format='sparse'))


sim = Simulator(ExampleTensorSummationSparse())
sim.run()

print('c', sim['c'].shape)
print(sim['c'])
print('einsum_summ2_sparse_derivs', sim['einsum_summ2_sparse_derivs'].shape)
print(sim['einsum_summ2_sparse_derivs'])
