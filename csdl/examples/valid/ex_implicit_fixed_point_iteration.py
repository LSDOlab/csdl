def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    from csdl.examples.models.quadratic_function import QuadraticFunction
    from csdl.examples.models.fixed_point import FixedPoint1, FixedPoint2, FixedPoint3
    from csdl.examples.models.fixed_point import FixedPoint2
    from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
    from csdl.examples.models.simple_add import SimpleAdd
    from csdl.examples.models.circle_parabola import CircleParabola
    from csdl.examples.models.quadratic_function import QuadraticFunction
    
    
    class ExampleFixedPointIteration(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.fixed_point import FixedPoint1, FixedPoint2, FixedPoint3
    
            solve_fixed_point_iteration1 = self.create_implicit_operation(
                FixedPoint1(name='a'))
            solve_fixed_point_iteration1.declare_state('a', residual='r')
            solve_fixed_point_iteration1.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            a = solve_fixed_point_iteration1()
    
            solve_fixed_point_iteration2 = self.create_implicit_operation(
                FixedPoint2(name='b'))
            solve_fixed_point_iteration2.declare_state('b', residual='r')
            solve_fixed_point_iteration2.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            b = solve_fixed_point_iteration2()
    
            solve_fixed_point_iteration3 = self.create_implicit_operation(
                FixedPoint3(name='c'))
            solve_fixed_point_iteration3.declare_state('c', residual='r')
            solve_fixed_point_iteration3.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            c = solve_fixed_point_iteration3()
    
    
    rep = GraphRepresentation(ExampleFixedPointIteration())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('b', sim['b'].shape)
    print(sim['b'])
    print('c', sim['c'].shape)
    print(sim['c'])
    
    return sim, rep