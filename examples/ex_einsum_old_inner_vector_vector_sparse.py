from csdl_om import Simulator
import numpy as np
from csdl import Model
import csdl


class ExampleInnerVectorVectorSparse(Model):
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        self.register_output(
            'einsum_inner1_sparse_derivs',
            csdl.einsum(
                vec,
                vec,
                subscripts='i,i->',
                partial_format='sparse',
            ))


sim = Simulator(ExampleInnerVectorVectorSparse())
sim.run()

print('a', sim['a'].shape)
print(sim['a'])
print('einsum_inner1_sparse_derivs', sim['einsum_inner1_sparse_derivs'].shape)
print(sim['einsum_inner1_sparse_derivs'])
