from csdl_om import Simulator
from csdl import Model, ImplicitModel, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np

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

sim = Simulator(ExampleWithSubsystemsVisualizeInternalModel())
sim.run()
