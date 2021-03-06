def example(Simulator):
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
    
    
    rep = GraphRepresentation(ExampleApplyNonlinearWithExpose())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep