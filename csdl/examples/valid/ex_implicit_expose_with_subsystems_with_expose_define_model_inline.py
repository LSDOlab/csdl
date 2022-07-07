def example(Simulator):
    import imp
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    from csdl.examples.models.fixed_point import FixedPoint2Expose
    import numpy as np
    
    
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
    
    
    rep = GraphRepresentation(ExampleWithSubsystemsWithExposeDefineModelInline())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep