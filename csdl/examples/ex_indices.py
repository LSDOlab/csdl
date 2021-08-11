import numpy as np

import csdl
from csdl import Model


class ExampleInteger(Model):
    """
    :param var: x
    :param var: x0
    :param var: x6
    """
    def define(self):
        a = self.declare_variable('a', val=0)
        b = self.declare_variable('b', val=1)
        c = self.declare_variable('c', val=2)
        d = self.declare_variable('d', val=7.4)
        e = self.declare_variable('e', val=np.pi)
        f = self.declare_variable('f', val=9)
        g = e + f
        x = self.create_output('x', shape=(7, ))
        x[0] = a
        x[1] = b
        x[2] = c
        x[3] = d
        x[4] = e
        x[5] = f
        x[6] = g

        # Get value from indices
        self.register_output('x0', x[0])
        self.register_output('x6', x[6])


class ErrorIntegerReuse(Model):
    def define(self):
        a = self.declare_variable('a', val=4)
        b = self.declare_variable('b', val=3)
        x = self.create_output('x', shape=(2, ))
        x[0] = a
        x[1] = b
        y = self.create_output('y', shape=(2, ))
        y[0] = x[0]
        y[1] = x[0]


class ExampleOneDimensional(Model):
    """
    :param var: x
    :param var: y
    :param var: z
    :param var: x0_5
    :param var: x3_
    :param var: x2_4
    """
    def define(self):
        n = 20
        u = self.declare_variable('u',
                                  shape=(n, ),
                                  val=np.arange(n).reshape((n, )))
        v = self.declare_variable('v',
                                  shape=(n - 4, ),
                                  val=np.arange(n - 4).reshape((n - 4, )))
        w = self.declare_variable('w',
                                  shape=(4, ),
                                  val=16 + np.arange(4).reshape((4, )))
        x = self.create_output('x', shape=(n, ))
        x[0:n] = 2 * (u + 1)
        y = self.create_output('y', shape=(n, ))
        y[0:n - 4] = 2 * (v + 1)
        y[n - 4:n] = w - 3

        # Get value from indices
        z = self.create_output('z', shape=(3, ))
        z[0:3] = csdl.expand(x[2], (3, ))
        self.register_output('x0_5', x[0:5])
        self.register_output('x3_', x[3:])
        self.register_output('x2_4', x[2:4])


class ErrorOneDimensionalReuse(Model):
    def define(self):
        n = 8
        u = self.declare_variable('u',
                                  shape=(n, ),
                                  val=np.arange(n).reshape((n, )))
        v = self.create_output('v', shape=(n, ))
        v[:4] = u[:4]
        v[4:] = u[:4]


class ExampleMultidimensional(Model):
    """
    :param var: x
    :param var: q
    :param var: r
    :param var: s
    :param var: t
    """
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


class ErrorIntegerOutOfRange(Model):
    def define(self):
        a = self.declare_variable('a', val=0)
        x = self.create_output('x', shape=(1, ))
        # This triggers an error
        x[1] = a


class ErrorIntegerOverlap(Model):
    def define(self):
        a = self.declare_variable('a', val=0)
        b = self.declare_variable('b', val=1)
        x = self.create_output('x', shape=(2, ))
        x[0] = a
        # This triggers an error
        x[0] = b


class ErrorOneDimensionalOutOfRange(Model):
    def define(self):
        n = 20
        x = self.declare_variable('x',
                                  shape=(n - 4, ),
                                  val=np.arange(n - 4).reshape((n - 4, )))
        y = self.declare_variable('y',
                                  shape=(4, ),
                                  val=16 + np.arange(4).reshape((4, )))
        z = self.create_output('z', shape=(n, ))
        z[0:n - 4] = 2 * (x + 1)
        # This triggers an error
        z[n - 3:n + 1] = y - 3


class ErrorOneDimensionalOverlap(Model):
    def define(self):
        n = 20
        x = self.declare_variable('x',
                                  shape=(n - 4, ),
                                  val=np.arange(n - 4).reshape((n - 4, )))
        y = self.declare_variable('y',
                                  shape=(4, ),
                                  val=16 + np.arange(4).reshape((4, )))
        z = self.create_output('z', shape=(n, ))
        z[0:n - 4] = 2 * (x + 1)
        # This triggers an error
        z[n - 5:n - 1] = y - 3


class ErrorMultidimensionalOutOfRange(Model):
    def define(self):
        z = self.declare_variable('z',
                                  shape=(2, 3),
                                  val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        # This triggers an error
        x[0:3, 0:3] = z


class ErrorMultidimensionalOverlap(Model):
    def define(self):
        z = self.declare_variable('z',
                                  shape=(2, 3),
                                  val=np.arange(6).reshape((2, 3)))
        x = self.create_output('x', shape=(2, 3))
        x[0:2, 0:3] = z
        # This triggers an error
        x[0:2, 0:3] = z


class ExampleMesh(Model):
    """
    :param var: def_mesh
    :param var: b_pts
    """
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
