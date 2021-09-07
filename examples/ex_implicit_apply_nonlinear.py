from csdl_om import Simulator
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


sim = Simulator(ExampleApplyNonlinear())
sim.run()
