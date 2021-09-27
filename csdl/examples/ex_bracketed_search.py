from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np


class ExampleBracketedScalar(Model):
    """
    :param var: x
    """
    def define(self):
        model = Model()
        a = model.declare_variable('a')
        b = model.declare_variable('b')
        c = model.declare_variable('c')
        x = model.declare_variable('x')
        y = a * x**2 + b * x + c
        model.register_output('y', y)

        a = self.declare_variable('a', val=1)
        b = self.declare_variable('b', val=-4)
        c = self.declare_variable('c', val=3)
        x = self.bracketed_search(
            a,
            b,
            c,
            states=['x'],
            residuals=['y'],
            model=model,
            brackets=dict(x=(0, 2)),
        )


class ExampleBracketedArray(Model):
    """
    :param var: x
    """
    def define(self):
        model = Model()
        a = model.declare_variable('a', shape=(2, ))
        b = model.declare_variable('b', shape=(2, ))
        c = model.declare_variable('c', shape=(2, ))
        x = model.declare_variable('x', shape=(2, ))
        y = a * x**2 + b * x + c
        model.register_output('y', y)

        a = model.declare_variable('a', val=[1, -1])
        b = model.declare_variable('b', val=[-4, 4])
        c = model.declare_variable('c', val=[3, -3])
        x = self.bracketed_search(
            a,
            b,
            c,
            states=['x'],
            residuals=['y'],
            model=model,
            brackets=dict(x=(
                np.array([0, 2.]),
                np.array([2, np.pi], ),
            )),
        )


class ExampleWithSubsystemsBracketedScalar(Model):
    """
    :param var: y
    """
    def define(self):
        with self.create_submodel('R') as model:
            p = model.create_input('p', val=7)
            q = model.create_input('q', val=8)
            r = p + q
            model.register_output('r', r)

        m2 = Model()
        x = m2.declare_variable('a')
        r = m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))

        m3 = Model()
        a = m3.declare_variable('a')
        b = m3.declare_variable('b')
        c = m3.declare_variable('c')
        r = m3.declare_variable('r')
        y = m3.declare_variable('y')
        m3.register_output('z', a * y**2 + b * y + c - r)

        a = self.implicit_operation(
            states=['a'],
            residuals=['r'],
            model=m2,
            nonlinear_solver=NewtonSolver(solve_subsystems=False),
            # nonlinear_solver=NonlinearBlockGS(maxiter=100),
            linear_solver=ScipyKrylov(),
        )

        b = self.declare_variable('b', val=-4)
        c = self.declare_variable('c', val=18)
        r = self.declare_variable('r')
        y = self.bracketed_search(
            a,
            b,
            c,
            r,
            model=m3,
            states=['y'],
            residuals=['z'],
            brackets=dict(y=(0, 2)),
        )


class ExampleWithSubsystemsBracketedArray(Model):
    """
    :param var: y
    """
    def define(self):
        with self.create_submodel('R') as model:
            p = model.create_input('p', val=[7, -7])
            q = model.create_input('q', val=[8, -8])
            r = p + q
            model.register_output('r', r)

        m2 = Model()
        x = m2.declare_variable('ap')
        r = m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))

        m3 = Model()
        a = m3.declare_variable('a', shape=(2, ))
        b = m3.declare_variable('b', shape=(2, ))
        c = m3.declare_variable('c', shape=(2, ))
        r = m3.declare_variable('r', shape=(2, ))
        y = m3.declare_variable('y', shape=(2, ))
        m3.register_output('z', a * y**2 + b * y + c - r)

        ap = self.implicit_operation(
            states=['ap'],
            residuals=['r'],
            model=m2,
            nonlinear_solver=NewtonSolver(solve_subsystems=False),
            # nonlinear_solver=NonlinearBlockGS(maxiter=100),
            linear_solver=ScipyKrylov(),
        )

        a = self.create_output('a', shape=(2, ))
        a[0] = ap
        a[1] = -ap

        b = self.declare_variable('b', val=[-4, 4])
        c = self.declare_variable('c', val=[18, -18])
        r = self.declare_variable('r', shape=(2, ))
        y = self.bracketed_search(
            a,
            b,
            c,
            r,
            model=m3,
            states=['y'],
            residuals=['z'],
            brackets=dict(y=(
                np.array([0, 2.]),
                np.array([2, np.pi], ),
            )),
        )
