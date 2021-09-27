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


class ExampleBracketedScalar(Model):
    """
    :param var: x
    """
    def define(self):
        with self.create_submodel('sys') as model:
            a = model.declare_variable('a', val=1)
            b = model.declare_variable('b', val=-4)
            c = model.declare_variable('c', val=3)

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
            brackets=dict(y=(0, 2)),
        )


class ExampleBracketedArray(Model):
    """
    :param var: x
    """
    def define(self):
        with self.create_submodel('sys') as model:
            a = model.declare_variable('a', val=[1, -1])
            b = model.declare_variable('b', val=[-4, 4])
            c = model.declare_variable('c', val=[3, -3])
            x = model.declare_variable('x', shape=(2, ))
            y = a * x**2 + b * x + c
            model.register_output('y', y)

        a = self.declare_variable('a', shape=(2, ))
        b = self.declare_variable('b', shape=(2, ))
        c = self.declare_variable('c', shape=(2, ))
        x = self.bracketed_search(
            a,
            b,
            c,
            states=['x'],
            residuals=['y'],
            model=model,
            brackets=dict(y=([0, 2.], [2, np.pi])),
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


class ExampleWithSubsystemsBracketedScalar(Model):
    """
    :param var: y
    """
    def define(self):
        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        self.add(model, name='R')
        # declare output of child system as input to parent system
        r = self.declare_variable('r')

        c = self.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with self.create_submodel('coeff_a') as model:
            a = model.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            model.nonlinear_solver = NonlinearBlockGS(iprint=0,
                                                      maxiter=100)

        a = self.declare_variable('a')

        with self.create_submodel('coeff_b') as model:
            model.create_input('b', val=-4)

        b = self.declare_variable('b')
        y = self.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(z, x1=0, x2=2)


class ExampleWithSubsystemsBracketedArray(Model):
    """
    :param var: y
    """
    def define(self):
        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=[7, -7])
        q = model.create_input('q', val=[8, -8])
        r = p + q
        model.register_output('r', r)

        # add child system
        self.add(model, name='R')
        # declare output of child system as input to parent system
        r = self.declare_variable('r', shape=(2, ))

        c = self.declare_variable('c', val=[18, -18])

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with self.create_submodel('coeff_a') as model:
            a = model.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            model.nonlinear_solver = NonlinearBlockGS(iprint=0,
                                                      maxiter=100)

        # store positive and negative values of `a` in an array
        ap = self.declare_variable('a')
        an = -ap
        a = self.create_output('vec_a', shape=(2, ))
        a[0] = ap
        a[1] = an

        with self.create_submodel('coeff_b') as model:
            model.create_input('b', val=[-4, 4])

        b = self.declare_variable('b', shape=(2, ))
        y = self.create_implicit_output('y', shape=(2, ))
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(
            z,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


class ExampleWithSubsystemsVisualizeInternalModel(Model):
    def define(self):
        self.visualize = True

        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        self.add(model, name='R')
        # declare output of child system as input to parent system
        r = self.declare_variable('r')

        c = self.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        model = Model()
        a = model.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        self.add(model, name='coeff_a')

        a = self.declare_variable('a')

        model = Model()
        model.create_input('b', val=-4)
        self.add(model, name='coeff_b')

        b = self.declare_variable('b')
        y = self.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual(z)

        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
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
