from csdl import Model, ImplicitModel, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np


class ExampleApplyNonlinear(ImplicitModel):
    def define(self):
        with self.create_model('sys') as model:
            model.create_input('a', val=1)
            model.create_input('b', val=-4)
            model.create_input('c', val=3)
        a = self.declare_variable('a')
        b = self.declare_variable('b')
        c = self.declare_variable('c')

        x = self.create_implicit_output('x')
        y = a * x**2 + b * x + c

        x.define_residual(y)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(solve_subsystems=False)


class ExampleBracketedScalar(ImplicitModel):
    """
    :param var: x
    """
    def define(self):
        with self.create_model('sys') as model:
            model.create_input('a', val=1)
            model.create_input('b', val=-4)
            model.create_input('c', val=3)
        a = self.declare_variable('a')
        b = self.declare_variable('b')
        c = self.declare_variable('c')

        x = self.create_implicit_output('x')
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=0,
            x2=2,
        )


class ExampleBracketedArray(ImplicitModel):
    """
    :param var: x
    """
    def define(self):
        with self.create_model('sys') as model:
            model.create_input('a', val=[1, -1])
            model.create_input('b', val=[-4, 4])
            model.create_input('c', val=[3, -3])
        a = self.declare_variable('a', shape=(2, ))
        b = self.declare_variable('b', shape=(2, ))
        c = self.declare_variable('c', shape=(2, ))

        x = self.create_implicit_output('x', shape=(2, ))
        y = a * x**2 + b * x + c

        x.define_residual_bracketed(
            y,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


class ExampleWithSubsystems(ImplicitModel):
    def define(self):
        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        self.add(model, name='R', promotes=['*'])
        # declare output of child system as input to parent system
        r = self.declare_variable('r')

        c = self.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        model = Model()
        a = model.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        self.add(model, name='coeff_a', promotes=['*'])

        a = self.declare_variable('a')

        model = Model()
        model.create_input('b', val=-4)
        self.add(model, name='coeff_b', promotes=['*'])

        b = self.declare_variable('b')
        y = self.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual(z)
        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=100,
        )


class ExampleWithSubsystemsBracketedScalar(ImplicitModel):
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
        self.add(model, name='R', promotes=['*'])
        # declare output of child system as input to parent system
        r = self.declare_variable('r')

        c = self.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with self.create_model('coeff_a') as model:
            a = model.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)

        a = self.declare_variable('a')

        with self.create_model('coeff_b') as model:
            model.create_input('b', val=-4)

        b = self.declare_variable('b')
        y = self.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(z, x1=0, x2=2)


class ExampleWithSubsystemsBracketedArray(ImplicitModel):
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
        self.add(model, name='R', promotes=['*'])
        # declare output of child system as input to parent system
        r = self.declare_variable('r', shape=(2, ))

        c = self.declare_variable('c', val=[18, -18])

        # a == (3 + a - 2 * a**2)**(1 / 4)
        with self.create_model('coeff_a') as model:
            a = model.create_output('a')
            a.define((3 + a - 2 * a**2)**(1 / 4))
            model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)

        # store positive and negative values of `a` in an array
        ap = self.declare_variable('a')
        an = -ap
        a = self.create_output('vec_a', shape=(2, ))
        a[0] = ap
        a[1] = an

        with self.create_model('coeff_b') as model:
            model.create_input('b', val=[-4, 4])

        b = self.declare_variable('b', shape=(2, ))
        y = self.create_implicit_output('y', shape=(2, ))
        z = a * y**2 + b * y + c - r
        y.define_residual_bracketed(
            z,
            x1=[0, 2.],
            x2=[2, np.pi],
        )


class ExampleWithSubsystems(ImplicitModel):
    def define(self):
        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        self.add(model, name='R', promotes=['*'])
        # declare output of child system as input to parent system
        r = self.declare_variable('r')

        c = self.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        model = Model()
        a = model.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        self.add(model, name='coeff_a', promotes=['*'])

        a = self.declare_variable('a')

        model = Model()
        model.create_input('b', val=-4)
        self.add(model, name='coeff_b', promotes=['*'])

        b = self.declare_variable('b')
        y = self.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual(z)


        # self.linear_solver = ScipyKrylov()
        # self.nonlinear_solver = NewtonSolver(
        #     solve_subsystems=False,
        #     maxiter=100,
        # )
class ExampleWithSubsystemsVisualizeInternalModel(ImplicitModel):
    def define(self):
        self.visualize = True

        # define a subsystem (this is a very simple example)
        model = Model()
        p = model.create_input('p', val=7)
        q = model.create_input('q', val=8)
        r = p + q
        model.register_output('r', r)

        # add child system
        self.add(model, name='R', promotes=['*'])
        # declare output of child system as input to parent system
        r = self.declare_variable('r')

        c = self.declare_variable('c', val=18)

        # a == (3 + a - 2 * a**2)**(1 / 4)
        model = Model()
        a = model.create_output('a')
        a.define((3 + a - 2 * a**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
        self.add(model, name='coeff_a', promotes=['*'])

        a = self.declare_variable('a')

        model = Model()
        model.create_input('b', val=-4)
        self.add(model, name='coeff_b', promotes=['*'])

        b = self.declare_variable('b')
        y = self.create_implicit_output('y')
        z = a * y**2 + b * y + c - r
        y.define_residual(z)
        # self.linear_solver = ScipyKrylov()
        # self.nonlinear_solver = NewtonSolver(
        #     solve_subsystems=False,
        #     maxiter=100,
        # )
