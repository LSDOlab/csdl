def example(Simulator):
    import imp
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    from csdl.examples.models.fixed_point import FixedPoint2Expose
    import numpy as np
    from csdl.examples.models.quadratic_function import QuadraticFunctionExpose
    from csdl.examples.models.fixed_point import FixedPoint1Expose, FixedPoint2Expose, FixedPoint3Expose
    from csdl.examples.models.simple_add import SimpleAdd
    from csdl.examples.models.quadratic_function import QuadraticFunctionExpose
    from csdl.examples.models.fixed_point import FixedPoint2Expose
    from csdl.examples.models.circle_parabola import CircleParabolaExpose
    
    
    class ExampleApplyNonlinearWithExposeDefineModelInline(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.quadratic_function import QuadraticFunctionExpose
    
            solve_quadratic = self.create_implicit_operation(
                QuadraticFunctionExpose(shape=(1, )))
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
    
    
    sim = Simulator(ExampleApplyNonlinearWithExposeDefineModelInline())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('t', sim['t'].shape)
    print(sim['t'])
    
    return sim