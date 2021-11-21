from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np


class ExampleApplyNonlinearWithExpose(Model):
    def define(self):
        # define internal model that defines a residual
        model = Model()
        a = model.declare_variable('a', val=1)
        b = model.declare_variable('b', val=-4)
        c = model.declare_variable('c', val=3)
        x = model.declare_variable('x')
        y = a * x**2 + b * x + c
        model.register_output('y', y)
        model.register_output('t', a + b + c)

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
        x, t = solve_quadratic(a, b, c, expose=['t'])


class ExampleFixedPointIterationWithExpose(Model):
    """
    :param var: a
    :param var: b
    :param var: c
    """
    def define(self):
        # x == (3 + x - 2 * x**2)**(1 / 4)
        m1 = Model()
        x = m1.declare_variable('a')
        m1.register_output('r', x - (3 + x - 2 * x**2)**(1 / 4))
        m1.register_output('t1', x**2)

        # x == ((x + 3 - x**4) / 2)**(1 / 4)
        m2 = Model()
        x = m2.declare_variable('b')
        m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))
        m2.register_output('t2', x**2)

        # x == 0.5 * x
        m3 = Model()
        x = m3.declare_variable('c')
        m3.register_output('r', x - 0.5 * x)

        solve_fixed_point_iteration1 = self.create_implicit_operation(
            m1)
        solve_fixed_point_iteration1.declare_state('a', residual='r')
        solve_fixed_point_iteration1.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a, t1 = solve_fixed_point_iteration1(expose=['t1'])

        solve_fixed_point_iteration2 = self.create_implicit_operation(
            m2)
        solve_fixed_point_iteration2.declare_state('b', residual='r')
        solve_fixed_point_iteration2.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        b, t2 = solve_fixed_point_iteration2(expose=['t2'])

        solve_fixed_point_iteration3 = self.create_implicit_operation(
            m3)
        solve_fixed_point_iteration3.declare_state('c', residual='r')
        solve_fixed_point_iteration3.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        c = solve_fixed_point_iteration3()


class ExampleWithSubsystemsWithExpose(Model):
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
        t2 = m2.register_output('t2', a**2)

        m3 = Model()
        a = m3.declare_variable('a')
        b = m3.declare_variable('b')
        c = m3.declare_variable('c')
        r = m3.declare_variable('r')
        y = m3.declare_variable('y')
        m3.register_output('z', a * y**2 + b * y + c - r)
        m3.register_output('t3', a + b + c - r)
        m3.register_output('t4', y**2)

        solve_fixed_point_iteration = self.create_implicit_operation(m2)
        solve_fixed_point_iteration.declare_state('a', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a, t2 = solve_fixed_point_iteration(expose=['t2'])

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
        y, t3, t4 = solve_quadratic(a, b, c, r, expose=['t3', 't4'])


class ExampleMultipleResidualsWithExpose(Model):
    """
    :param var: x
    :param var: y
    :param var: t1
    :param var: t2
    :param var: t3
    :param var: t4
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
        m.register_output('t1', a + b + c)
        m.register_output('t2', x**2)
        m.register_output('t3', 2 * y)
        m.register_output('t4', x + y)

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

        x, y, t1, t2, t3, t4 = solve_multiple_implicit(
            r,
            a,
            b,
            c,
            expose=['t1', 't2', 't3', 't4'],
        )
