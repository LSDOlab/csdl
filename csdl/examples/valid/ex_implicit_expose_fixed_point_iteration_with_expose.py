def example(Simulator):
    import imp
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    from csdl.examples.models.fixed_point import FixedPoint2Expose
    import numpy as np
    from csdl.examples.models.quadratic_function import QuadraticFunctionExpose
    from csdl.examples.models.fixed_point import FixedPoint1Expose, FixedPoint2Expose, FixedPoint3Expose
    from csdl.examples.models.simple_add import SimpleAdd
    from csdl.examples.models.quadratic_function import QuadraticFunctionExpose
    from csdl.examples.models.fixed_point import FixedPoint2Expose
    from csdl.examples.models.circle_parabola import CircleParabolaExpose
    
    
    class ExampleFixedPointIterationWithExpose(Model):
        def define(self):
            # x == (3 + x - 2 * x**2)**(1 / 4)
            m1 = Model()
            x = m1.declare_variable('a')
            m1.register_output('r', x - (3 + x - 2 * x**2)**(1 / 4))
            m1.register_output('t1', x**2)
    
            # x == ((x + 3 - x**4) / 2)**(1 / 4)
            m2 = Model()
            x = m2.declare_variable('b')
            m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))
            m2.register_output('t2', x**2)
    
            # x == 0.5 * x
            m3 = Model()
            x = m3.declare_variable('c')
            m3.register_output('r', x - 0.5 * x)
    
            solve_fixed_point_iteration1 = self.create_implicit_operation(
                m1)
            solve_fixed_point_iteration1.declare_state('a', residual='r')
            solve_fixed_point_iteration1.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            a, t1 = solve_fixed_point_iteration1(expose=['t1'])
    
            solve_fixed_point_iteration2 = self.create_implicit_operation(
                m2)
            solve_fixed_point_iteration2.declare_state('b', residual='r')
            solve_fixed_point_iteration2.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            b, t2 = solve_fixed_point_iteration2(expose=['t2'])
    
            solve_fixed_point_iteration3 = self.create_implicit_operation(
                m3)
            solve_fixed_point_iteration3.declare_state('c', residual='r')
            solve_fixed_point_iteration3.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            c = solve_fixed_point_iteration3()
    
    
    rep = GraphRepresentation(ExampleFixedPointIterationWithExpose())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('b', sim['b'].shape)
    print(sim['b'])
    print('c', sim['c'].shape)
    print(sim['c'])
    
    return sim, rep