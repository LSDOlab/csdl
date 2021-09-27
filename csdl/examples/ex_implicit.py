from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np


class ExampleApplyNonlinear(Model):
    def define(self):
        # define internal model that defines a residual
        model = Model()
        a = model.declare_variable('a', val=1)
        b = model.declare_variable('b', val=-4)
        c = model.declare_variable('c', val=3)
        x = model.declare_variable('x')
        y = a * x**2 + b * x + c
        model.register_output('y', y)

        # define arguments to implicit operation
        a = self.declare_variable('a', val=1)
        b = self.declare_variable('b', val=-4)
        c = self.declare_variable('c', val=3)

        # define output of implicit operation
        x = self.implicit_operation(
            a,
            b,
            c,
            states=['x'],
            residuals=['y'],
            model=model,
            nonlinear_solver=NewtonSolver(solve_subsystems=False),
            linear_solver=ScipyKrylov(),
        )


class ExampleFixedPointIteration(Model):
    """
    :param var: a
    :param var: b
    :param var: c
    """
    def define(self):
        # x == (3 + x - 2 * x**2)**(1 / 4)
        m1 = Model()
        x = m1.declare_variable('a')
        r = m1.register_output('r', x - (3 + x - 2 * x**2)**(1 / 4))
        a = self.implicit_operation(
            states=['a'],
            residuals=['r'],
            model=m1,
            nonlinear_solver=NewtonSolver(solve_subsystems=False),
            # nonlinear_solver=NonlinearBlockGS(maxiter=100),
            linear_solver=ScipyKrylov(),
        )

        # x == ((x + 3 - x**4) / 2)**(1 / 4)
        m2 = Model()
        x = m2.declare_variable('b')
        r = m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))
        b = self.implicit_operation(
            states=['b'],
            residuals=['r'],
            model=m2,
            nonlinear_solver=NewtonSolver(solve_subsystems=False),
            # nonlinear_solver=NonlinearBlockGS(maxiter=100),
            linear_solver=ScipyKrylov(),
        )

        # x == 0.5 * x
        m3 = Model()
        x = m3.declare_variable('c')
        r = m3.register_output('r', x - 0.5 * x)
        c = self.implicit_operation(
            states=['c'],
            residuals=['r'],
            model=m3,
            nonlinear_solver=NewtonSolver(solve_subsystems=False),
            # nonlinear_solver=NonlinearBlockGS(maxiter=100),
            linear_solver=ScipyKrylov(),
        )


class ExampleWithSubsystems(Model):
    def define(self):
        with self.create_submodel('R') as model:
            p = model.create_input('p', val=7)
            q = model.create_input('q', val=8)
            r = p + q
            model.register_output('r', r)
        r = self.declare_variable('r')

        m2 = Model()
        a = m2.declare_variable('a')
        m2.register_output('r', a - (3 + a - 2 * a**2)**(1 / 4))

        # x == ((x + 3 - x**4) / 2)**(1 / 4)
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
            nonlinear_solver=NewtonSolver(
                solve_subsystems=False,
                maxiter=100,
                iprint=False,
            ),
            # nonlinear_solver=NonlinearBlockGS(maxiter=100),
            linear_solver=ScipyKrylov(),
        )

        b = self.create_input('b', val=-4)
        c = self.declare_variable('c', val=18)
        y = self.implicit_operation(
            a,
            b,
            c,
            r,
            states=['y'],
            residuals=['z'],
            model=m3,
            nonlinear_solver=NewtonSolver(
                solve_subsystems=False,
                maxiter=100,
                iprint=False,
            ),
            linear_solver=ScipyKrylov(),
        )


class ExampleCompositeResidual(Model):
    """
    :param var: x
    :param var: y
    """
    def define(self):
        m = Model()
        r = m.declare_variable('r')
        a = m.declare_variable('a')
        b = m.declare_variable('b')
        c = m.declare_variable('c')
        x = m.declare_variable('x', val=1.5)
        y = m.declare_variable('y', val=0.9)
        m.register_output('rx', x**2 + (y - r)**2 - r**2)
        m.register_output('ry', a * y**2 + b * y + c)

        r = self.declare_variable('r', val=2)
        a = self.declare_variable('a', val=1)
        b = self.declare_variable('b', val=-3)
        c = self.declare_variable('c', val=2)
        x, y = self.implicit_operation(
            r,
            a,
            b,
            c,
            model=m,
            states=['x', 'y'],
            residuals=['rx', 'ry'],
            linear_solver=ScipyKrylov(),
            nonlinear_solver=NewtonSolver(solve_subsystems=False),
        )
