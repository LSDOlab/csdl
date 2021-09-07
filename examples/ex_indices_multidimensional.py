from csdl_om import Simulator
import numpy as np
import csdl
from csdl import Model


class ExampleMultidimensional(Model):
    def define(self):
        # Works with two dimensional arrays
        z = self.declare_variable('z',
                                  shape=(2, 3),
                                  val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        x[0:2, 0:3] = z

        # Also works with higher dimensional arrays
        p = self.declare_variable('p',
                                  shape=(5, 2, 3),
                                  val=np.arange(30).reshape((5, 2, 3)))
        q = self.create_output('q', shape=(5, 2, 3))
        q[0:5, 0:2, 0:3] = p

        # Get value from indices
        self.register_output('r', p[0, :, :])

        # Assign a vector to a slice
        vec = self.create_input(
            'vec',
            shape=(1, 20),
            val=np.arange(20).reshape((1, 20)),
        )
        s = self.create_output('s', shape=(2, 20))
        s[0, :] = vec
        s[1, :] = 2 * vec

        # Negative indices
        t = self.create_output('t', shape=(5, 3, 3), val=0)
        t[0:5, 0:-1, 0:3] = p


sim = Simulator(ExampleMultidimensional())
sim.run()

print('x', sim['x'].shape)
print(sim['x'])
print('q', sim['q'].shape)
print(sim['q'])
print('r', sim['r'].shape)
print(sim['r'])
print('s', sim['s'].shape)
print(sim['s'])
print('t', sim['t'].shape)
print(sim['t'])
