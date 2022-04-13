from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np


class ExampleApplyNonlinear(Model):
    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.quadratic_function import QuadraticFunction

        solve_quadratic = self.create_implicit_operation(
            QuadraticFunction(shape=(1, )))
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
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.fixed_point import FixedPoint1, FixedPoint2, FixedPoint3

        solve_fixed_point_iteration1 = self.create_implicit_operation(
            FixedPoint1(name='a'))
        solve_fixed_point_iteration1.declare_state('a', residual='r')
        solve_fixed_point_iteration1.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a = solve_fixed_point_iteration1()

        solve_fixed_point_iteration2 = self.create_implicit_operation(
            FixedPoint2(name='b'))
        solve_fixed_point_iteration2.declare_state('b', residual='r')
        solve_fixed_point_iteration2.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        b = solve_fixed_point_iteration2()

        solve_fixed_point_iteration3 = self.create_implicit_operation(
            FixedPoint3(name='c'))
        solve_fixed_point_iteration3.declare_state('c', residual='r')
        solve_fixed_point_iteration3.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        c = solve_fixed_point_iteration3()


class ExampleWithSubsystems(Model):
    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.fixed_point import FixedPoint2
        from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
        from csdl.examples.models.simple_add import SimpleAdd

        self.add(SimpleAdd(p=7, q=8), name='R')
        r = self.declare_variable('r')

        solve_fixed_point_iteration = self.create_implicit_operation(
            FixedPoint2(name='a'))
        solve_fixed_point_iteration.declare_state('a', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)
        a = solve_fixed_point_iteration()

        solve_quadratic = self.create_implicit_operation(
            QuadraticWithExtraTerm(shape=(1, )))
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
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.circle_parabola import CircleParabola
        r = self.declare_variable('r', val=2)
        a = self.declare_variable('a', val=1)
        b = self.declare_variable('b', val=-3)
        c = self.declare_variable('c', val=2)
        solve_multiple_implicit = self.create_implicit_operation(
            CircleParabola())
        solve_multiple_implicit.declare_state('x',
                                              residual='rx',
                                              val=1.5)
        solve_multiple_implicit.declare_state('y',
                                              residual='ry',
                                              val=0.9)
        solve_multiple_implicit.linear_solver = ScipyKrylov()
        solve_multiple_implicit.nonlinear_solver = NewtonSolver(
            solve_subsystems=False)

        x, y = solve_multiple_implicit(r, a, b, c)


class ExampleApplyNonlinearDefineModelInline(Model):
    def define(self):
        from csdl.examples.models.quadratic_function import QuadraticFunction

        solve_quadratic = self.create_implicit_operation(
            QuadraticFunction(shape=(1, )))
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


class ExampleFixedPointIterationDefineModelInline(Model):
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


class ExampleWithSubsystemsDefineModelInline(Model):
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


class ExampleMultipleResidualsDefineModelInline(Model):
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
