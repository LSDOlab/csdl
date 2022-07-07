def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
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
    
    
    rep = GraphRepresentation(ExampleApplyNonlinearDefineModelInline())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep