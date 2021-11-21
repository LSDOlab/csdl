import numpy as np
from csdl import Model
import csdl


class ExampleRegisterDependencies(Model):
    def define(self):
        a = self.declare_variable('a')
        b = self.declare_variable('b')
        c = self.declare_variable('c')
        d = self.declare_variable('d')
        x = a + b  # 2
        y = c + d  # 2
        z = x * y  # 4
        self.register_output('z', z)
        self.register_output('x', x)
        self.register_output('y', y)


from csdl_om import Simulator

sim = Simulator(ExampleRegisterDependencies())
sim.run()
print(sim['z'])
print(sim['x'])
print(sim['y'])
