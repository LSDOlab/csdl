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

        solve_quadratic = self.create_implicit_operation(model)
        solve_quadratic.declare_state('x', residual='y')
        solve_quadratic.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )
        solve_quadratic.linear_solver = ScipyKrylov()

        a = self.declare_variable('a', val=1)
        b = self.declare_variable('b', val=-4)
        c = self.declare_variable('c', val=3)
        x = solve_quadratic(a, b, c)


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

        # x == ((x + 3 - x**4) / 2)**(1 / 4)
        m2 = Model()
        x = m2.declare_variable('b')
        r = m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))

        # x == 0.5 * x
        m3 = Model()
        x = m3.declare_variable('c')
        r = m3.register_output('r', x - 0.5 * x)

        solve_fixed_point_iteration1 = self.create_implicit_operation(
            m1)
        solve_fixed_point_iteration1.declare_state('a', residual='r')
        solve_fixed_point_iteration1.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a = solve_fixed_point_iteration1()

        solve_fixed_point_iteration2 = self.create_implicit_operation(
            m2)
        solve_fixed_point_iteration2.declare_state('b', residual='r')
        solve_fixed_point_iteration2.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        b = solve_fixed_point_iteration2()

        solve_fixed_point_iteration3 = self.create_implicit_operation(
            m3)
        solve_fixed_point_iteration3.declare_state('c', residual='r')
        solve_fixed_point_iteration3.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        c = solve_fixed_point_iteration3()


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
        r = m2.register_output('r', a - ((a + 3 - a**4) / 2)**(1 / 4))

        m3 = Model()
        a = m3.declare_variable('a')
        b = m3.declare_variable('b')
        c = m3.declare_variable('c')
        r = m3.declare_variable('r')
        y = m3.declare_variable('y')
        m3.register_output('z', a * y**2 + b * y + c - r)

        solve_fixed_point_iteration = self.create_implicit_operation(m2)
        solve_fixed_point_iteration.declare_state('a', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a = solve_fixed_point_iteration()

        solve_quadratic = self.create_implicit_operation(m3)
        b = self.create_input('b', val=-4)
        solve_quadratic.declare_state('y', residual='z')
        solve_quadratic.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )
        solve_quadratic.linear_solver = ScipyKrylov()

        c = self.declare_variable('c', val=18)
        y = solve_quadratic(a, b, c, r)


class ExampleMultipleResiduals(Model):
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
        solve_multiple_implicit = self.create_implicit_operation(m)
        solve_multiple_implicit.declare_state('x', residual='rx')
        solve_multiple_implicit.declare_state('y', residual='ry')
        solve_multiple_implicit.linear_solver = ScipyKrylov()
        solve_multiple_implicit.nonlinear_solver = NewtonSolver(
            solve_subsystems=False)

        x, y = solve_multiple_implicit(r, a, b, c)
