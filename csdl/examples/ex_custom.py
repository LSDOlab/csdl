from csdl import CustomExplicitOperation, CustomImplicitOperation, NewtonSolver, ScipyKrylov
import csdl
import numpy as np


class ExampleImplicitSimple(CustomImplicitOperation):
    """
    :param var: x
    """
    def define(self):
        self.add_input('a', val=1.)
        self.add_input('b', val=-4.)
        self.add_input('c', val=3.)
        self.add_output('x', val=0.)
        self.declare_derivatives('x', 'x')
        self.declare_derivatives('x', ['a', 'b', 'c'])

        self.linear_solver = ScipyKrylov()
        self.nonlinear_solver = NewtonSolver(solve_subsystems=False)

    def evaluate_residuals(self, inputs, outputs, residuals):
        x = outputs['x']
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        residuals['x'] = a * x**2 + b * x + c

    def compute_derivatives(self, inputs, outputs, derivatives):
        a = inputs['a']
        b = inputs['b']
        x = outputs['x']

        derivatives['x', 'a'] = x**2
        derivatives['x', 'b'] = x
        derivatives['x', 'c'] = 1.0
        derivatives['x', 'x'] = 2 * a * x + b
