from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np


class ExampleBracketedScalar(Model):
    """
    :param var: x
    """
    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.quadratic_function import QuadraticFunction

        solve_quadratic = self.create_implicit_operation(
            QuadraticFunction(shape=(1, )))
        solve_quadratic.declare_state('x', residual='y', bracket=(0, 2))

        a = self.declare_variable('a', val=1)
        b = self.declare_variable('b', val=-4)
        c = self.declare_variable('c', val=3)
        x = solve_quadratic(a, b, c)


class ExampleBracketedArray(Model):
    """
    :param var: x
    """
    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.quadratic_function import QuadraticFunction
        solve_quadratic = self.create_implicit_operation(
            QuadraticFunction(shape=(2, )))
        solve_quadratic.declare_state('x',
                                      residual='y',
                                      bracket=(
                                          np.array([0, 2.]),
                                          np.array([2, np.pi], ),
                                      ))

        a = self.declare_variable('a', val=[1, -1])
        b = self.declare_variable('b', val=[-4, 4])
        c = self.declare_variable('c', val=[3, -3])
        x = solve_quadratic(a, b, c)


class ExampleWithSubsystemsBracketedScalar(Model):
    """
    :param var: y
    """
    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
        from csdl.examples.models.simple_add import SimpleAdd
        from csdl.examples.models.fixed_point import FixedPoint2
        self.add(SimpleAdd(p=7, q=8), name='R')
        solve_fixed_point_iteration = self.create_implicit_operation(
            FixedPoint2(name='a'))
        solve_fixed_point_iteration.declare_state('a', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)

        solve_quadratic = self.create_implicit_operation(
            QuadraticWithExtraTerm(shape=(1, )))
        solve_quadratic.declare_state('y', residual='z', bracket=(0, 2))
        solve_quadratic.nonlinear_solver = NonlinearBlockGS(maxiter=100)

        a = solve_fixed_point_iteration()

        b = self.declare_variable('b', val=-4)
        c = self.declare_variable('c', val=18)
        r = self.declare_variable('r')
        y = solve_quadratic(a, b, c, r)


class ExampleWithSubsystemsBracketedArray(Model):
    """
    :param var: y
    """
    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
        from csdl.examples.models.simple_add import SimpleAdd
        from csdl.examples.models.fixed_point import FixedPoint2
        self.add(SimpleAdd(p=[7, -7], q=[8, -8]), name='R')

        solve_fixed_point_iteration = self.create_implicit_operation(
            FixedPoint2(name='ap'))
        solve_fixed_point_iteration.declare_state('ap', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)

        solve_quadratic = self.create_implicit_operation(
            QuadraticWithExtraTerm(shape=(2, )))
        solve_quadratic.declare_state('y',
                                      residual='z',
                                      bracket=(
                                          np.array([0, 2.]),
                                          np.array([2, np.pi], ),
                                      ))
        solve_quadratic.nonlinear_solver = NonlinearBlockGS(maxiter=100)

        ap = solve_fixed_point_iteration()
        a = self.create_output('a', shape=(2, ))
        a[0] = ap
        a[1] = -ap

        b = self.declare_variable('b', val=[-4, 4])
        c = self.declare_variable('c', val=[18, -18])
        r = self.declare_variable('r', shape=(2, ))
        y = solve_quadratic(a, b, c, r)


# ----------------------------------------------------------------------
class ExampleBracketedScalarDefineModelInline(Model):
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

        solve_quadratic = self.create_implicit_operation(model)
        solve_quadratic.declare_state('x', residual='y', bracket=(0, 2))

        a = self.declare_variable('a', val=1)
        b = self.declare_variable('b', val=-4)
        c = self.declare_variable('c', val=3)
        x = solve_quadratic(a, b, c)


class ExampleBracketedArrayDefineModelInline(Model):
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

        solve_quadratic = self.create_implicit_operation(model)
        solve_quadratic.declare_state('x',
                                      residual='y',
                                      bracket=(
                                          np.array([0, 2.]),
                                          np.array([2, np.pi], ),
                                      ))

        a = self.declare_variable('a', val=[1, -1])
        b = self.declare_variable('b', val=[-4, 4])
        c = self.declare_variable('c', val=[3, -3])
        x = solve_quadratic(a, b, c)


class ExampleWithSubsystemsBracketedScalarDefineModelInline(Model):
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

        solve_fixed_point_iteration = self.create_implicit_operation(m2)
        solve_fixed_point_iteration.declare_state('a', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)

        solve_quadratic = self.create_implicit_operation(m3)
        solve_quadratic.declare_state('y', residual='z', bracket=(0, 2))
        solve_quadratic.nonlinear_solver = NonlinearBlockGS(maxiter=100)

        a = solve_fixed_point_iteration()

        b = self.declare_variable('b', val=-4)
        c = self.declare_variable('c', val=18)
        r = self.declare_variable('r')
        y = solve_quadratic(a, b, c, r)


class ExampleWithSubsystemsBracketedArrayDefineModelInline(Model):
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

        solve_fixed_point_iteration = self.create_implicit_operation(m2)
        solve_fixed_point_iteration.declare_state('ap', residual='r')
        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
            maxiter=100)

        solve_quadratic = self.create_implicit_operation(m3)
        solve_quadratic.declare_state('y',
                                      residual='z',
                                      bracket=(
                                          np.array([0, 2.]),
                                          np.array([2, np.pi], ),
                                      ))
        solve_quadratic.nonlinear_solver = NonlinearBlockGS(maxiter=100)

        ap = solve_fixed_point_iteration()
        a = self.create_output('a', shape=(2, ))
        a[0] = ap
        a[1] = -ap

        b = self.declare_variable('b', val=[-4, 4])
        c = self.declare_variable('c', val=[18, -18])
        r = self.declare_variable('r', shape=(2, ))
        y = solve_quadratic(a, b, c, r)
