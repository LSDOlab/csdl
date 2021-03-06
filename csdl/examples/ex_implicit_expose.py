import imp
from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
from csdl.examples.models.fixed_point import FixedPoint2Expose
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
        x = m3.declare_variable('x')
        m3.register_output('y', a * x**2 + b * x + c - r)
        m3.register_output('t3', a + b + c - r)
        m3.register_output('t4', x**2)

        solve_fixed_point_iteration = self.create_implicit_operation(m2)
        solve_fixed_point_iteration.declare_state('a', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a, t2 = solve_fixed_point_iteration(expose=['t2'])

        solve_quadratic = self.create_implicit_operation(m3)
        b = self.create_input('b', val=-4)
        solve_quadratic.declare_state('x', residual='y')
        solve_quadratic.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )
        solve_quadratic.linear_solver = ScipyKrylov()

        c = self.declare_variable('c', val=18)
        x, t3, t4 = solve_quadratic(a, b, c, r, expose=['t3', 't4'])


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


# ----------------------------------------------------------------------


class ExampleApplyNonlinearWithExposeDefineModelInline(Model):
    """
    :param var: x
    :param var: t
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.quadratic_function import QuadraticFunctionExpose

        solve_quadratic = self.create_implicit_operation(
            QuadraticFunctionExpose(shape=(1, )))
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


class ExampleFixedPointIterationWithExposeDefineModelInline(Model):
    """
    :param var: a
    :param var: b
    :param var: c
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.fixed_point import FixedPoint1Expose, FixedPoint2Expose, FixedPoint3Expose

        solve_fixed_point_iteration1 = self.create_implicit_operation(
            FixedPoint1Expose(name='a'))
        solve_fixed_point_iteration1.declare_state('a', residual='r')
        solve_fixed_point_iteration1.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a, t1 = solve_fixed_point_iteration1(expose=['t1'])

        solve_fixed_point_iteration2 = self.create_implicit_operation(
            FixedPoint2Expose(name='b'))
        solve_fixed_point_iteration2.declare_state('b', residual='r')
        solve_fixed_point_iteration2.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        b, t2 = solve_fixed_point_iteration2(expose=['t2'])

        solve_fixed_point_iteration3 = self.create_implicit_operation(
            FixedPoint3Expose(name='c'))
        solve_fixed_point_iteration3.declare_state('c', residual='r')
        solve_fixed_point_iteration3.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        c = solve_fixed_point_iteration3()


class ExampleWithSubsystemsWithExposeDefineModelInline(Model):

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.simple_add import SimpleAdd
        from csdl.examples.models.quadratic_function import QuadraticFunctionExpose
        from csdl.examples.models.fixed_point import FixedPoint2Expose

        self.add(SimpleAdd(p=7, q=8), name='R')
        r = self.declare_variable('r')

        solve_fixed_point_iteration = self.create_implicit_operation(
            FixedPoint2Expose(name='a'))
        solve_fixed_point_iteration.declare_state('a', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a, t2 = solve_fixed_point_iteration(expose=['t2'])

        solve_quadratic = self.create_implicit_operation(
            QuadraticFunctionExpose(shape=(1, )))
        b = self.create_input('b', val=-4)
        solve_quadratic.declare_state('x', residual='y')
        solve_quadratic.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
            iprint=False,
        )
        solve_quadratic.linear_solver = ScipyKrylov()

        c = self.declare_variable('c', val=18)
        x, t3, t4, y = solve_quadratic(
            a,
            b,
            c,
            r,
            expose=['t3', 't4', 'y'],
        )


class ExampleMultipleResidualsWithExposeDefineModelInline(Model):
    """
    :param var: x
    :param var: y
    :param var: t1
    :param var: t2
    :param var: t3
    :param var: t4
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.circle_parabola import CircleParabolaExpose

        solve_multiple_implicit = self.create_implicit_operation(
            CircleParabolaExpose())
        solve_multiple_implicit.declare_state('x', residual='rx')
        solve_multiple_implicit.declare_state('y', residual='ry')
        solve_multiple_implicit.linear_solver = ScipyKrylov()
        solve_multiple_implicit.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
        )

        r = self.declare_variable('r', val=2)
        a = self.declare_variable('a', val=1)
        b = self.declare_variable('b', val=-3)
        c = self.declare_variable('c', val=2)
        x, y, t1, t2, t3, t4 = solve_multiple_implicit(
            r,
            a,
            b,
            c,
            expose=['t1', 't2', 't3', 't4'],
        )
