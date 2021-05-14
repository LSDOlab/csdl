from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl import Model, ImplicitComponent
import numpy as np


class ExampleApplyNonlinear(ImplicitComponent):
    def define(self):
        m = self.model

        with m.create_model('sys') as model:
            model.create_input('a', val=1)
            model.create_input('b', val=-4)
            model.create_input('c', val=3)
        a = m.declare_variable('a')
        b = m.declare_variable('b')
        c = m.declare_variable('c')

        x = m.create_implicit_output('x')
        y = a * x**2 + b * x + c

        x.define_residual(y)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(solve_subsystems=False)


class ExampleBracketedScalar(ImplicitComponent):
    """
    :param var: x
    """
    def define(self):
        m = self.model

        with m.create_model('sys') as model:
            model.create_input('a', val=1)
            model.create_input('b', val=-4)
            model.create_input('c', val=3)
        a = m.declare_variable('a')
        b = m.declare_variable('b')
        c = m.declare_variable('c')

        x = m.create_implicit_output('x')
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=0,
            x2=2,
        )


class ExampleBracketedArray(ImplicitComponent):
    """
    :param var: x
    """
    def define(self):
        m = self.model

        with m.create_model('sys') as model:
            model.create_input('a', val=[1, -1])
            model.create_input('b', val=[-4, 4])
            model.create_input('c', val=[3, -3])
        a = m.declare_variable('a', shape=(2, ))
        b = m.declare_variable('b', shape=(2, ))
        c = m.declare_variable('c', shape=(2, ))

        x = m.create_implicit_output('x', shape=(2, ))
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


class ExampleWithSubsystems(ImplicitComponent):
    def define(self):
        m = self.model

        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        m.add(model, promotes=['*'])
        # declare output of child system as input to parent system
        r = m.declare_variable('r')

        c = m.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        model = Model()
        a = model.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        m.add('coeff_a', model, promotes=['*'])

        a = m.declare_variable('a')

        model = Model()
        model.create_input('b', val=-4)
        m.add('coeff_b', model, promotes=['*'])

        b = m.declare_variable('b')
        y = m.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual(z)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
        )


class ExampleWithSubsystemsBracketedScalar(ImplicitComponent):
    """
    :param var: y
    """
    def define(self):
        m = self.model

        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        m.add('R', model, promotes=['*'])
        # declare output of child system as input to parent system
        r = m.declare_variable('r')

        c = m.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with m.create_model('coeff_a') as model:
            a = model.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)

        a = m.declare_variable('a')

        with m.create_model('coeff_b') as model:
            model.create_input('b', val=-4)

        b = m.declare_variable('b')
        y = m.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(z, x1=0, x2=2)


class ExampleWithSubsystemsBracketedArray(ImplicitComponent):
    """
    :param var: y
    """
    def define(self):
        m = self.model

        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=[7, -7])
        q = model.create_input('q', val=[8, -8])
        r = p + q
        model.register_output('r', r)

        # add child system
        m.add('R', model, promotes=['*'])
        # declare output of child system as input to parent system
        r = m.declare_variable('r', shape=(2, ))

        c = m.declare_variable('c', val=[18, -18])

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with m.create_model('coeff_a') as model:
            a = model.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)

        # store positive and negative values of `a` in an array
        ap = m.declare_variable('a')
        an = -ap
        a = m.create_output('vec_a', shape=(2, ))
        a[0] = ap
        a[1] = an

        with m.create_model('coeff_b') as model:
            model.create_input('b', val=[-4, 4])

        b = m.declare_variable('b', shape=(2, ))
        y = m.create_implicit_output('y', shape=(2, ))
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(
            z,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


class ExampleWithSubsystemsInternalN2(ImplicitComponent):
    """
    :param option: n2=True
    """
    def define(self):
        m = self.model

        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        m.add('R', model, promotes=['*'])
        # declare output of child system as input to parent system
        r = m.declare_variable('r')

        c = m.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        model = Model()
        a = model.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        m.add('coeff_a', model, promotes=['*'])

        a = m.declare_variable('a')

        model = Model()
        model.create_input('b', val=-4)
        m.add('coeff_b', model, promotes=['*'])

        b = m.declare_variable('b')
        y = m.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual(z)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
        )
