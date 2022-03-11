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
    
    
    class ExampleFixedPointIterationWithExposeDefineModelInline(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.fixed_point import FixedPoint1Expose, FixedPoint2Expose, FixedPoint3Expose
    
            solve_fixed_point_iteration1 = self.create_implicit_operation(
                FixedPoint1Expose(name='a'))
            solve_fixed_point_iteration1.declare_state('a', residual='r')
            solve_fixed_point_iteration1.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            a, t1 = solve_fixed_point_iteration1(expose=['t1'])
    
            solve_fixed_point_iteration2 = self.create_implicit_operation(
                FixedPoint2Expose(name='b'))
            solve_fixed_point_iteration2.declare_state('b', residual='r')
            solve_fixed_point_iteration2.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            b, t2 = solve_fixed_point_iteration2(expose=['t2'])
    
            solve_fixed_point_iteration3 = self.create_implicit_operation(
                FixedPoint3Expose(name='c'))
            solve_fixed_point_iteration3.declare_state('c', residual='r')
            solve_fixed_point_iteration3.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            c = solve_fixed_point_iteration3()
    
    
    sim = Simulator(ExampleFixedPointIterationWithExposeDefineModelInline())
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('b', sim['b'].shape)
    print(sim['b'])
    print('c', sim['c'].shape)
    print(sim['c'])
    
    return sim