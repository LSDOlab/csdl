from csdl_om import Simulator
import numpy as np
import csdl
from csdl import Model

class ExampleMesh(Model):
    def define(self):
        nx = 3
        ny = 4
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0.
        def_mesh = self.declare_variable('def_mesh', val=mesh)
        b_pts = def_mesh[:-1, :, :] * .75 + def_mesh[1:, :, :] * .25
        self.register_output('b_pts', b_pts)

sim = Simulator(ExampleMesh())
sim.run()

print('def_mesh', sim['def_mesh'].shape)
print(sim['def_mesh'])
print('b_pts', sim['b_pts'].shape)
print(sim['b_pts'])
