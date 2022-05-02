from csdl import Model, GraphRepresentation
import csdl
import numpy as np


class ExampleSimple(Model):
    """
    :param var: quarter_chord
    :param var: widths
    """

    def define(self):
        # add_input
        nx = 3
        ny = 4
        mesh = np.zeros((nx, ny, 3))

        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0.
        def_mesh = self.declare_variable('def_mesh', val=mesh)

        # compute_output
        quarter_chord = def_mesh[nx -
                                 1, :, :] * 0.25 + def_mesh[0, :, :] * 0.75
        b_pts = def_mesh[:-1, :, :] * .75 + def_mesh[1:, :, :] * .25
        self.register_output('quarter_chord', quarter_chord)
        self.register_output('b_pts', b_pts)

        quarter_chord = self.declare_variable(
            'quarter_chord',
            val=np.ones((1, 4, 3)),
        )
        e = quarter_chord[:, :nx, :]
        f = quarter_chord[:, 1:, :]

        # this will combine operations
        widths = csdl.pnorm(f - e)
        self.register_output('widths', widths)
