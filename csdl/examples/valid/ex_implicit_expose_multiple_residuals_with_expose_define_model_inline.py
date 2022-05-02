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
    
    
    class ExampleMultipleResidualsWithExposeDefineModelInline(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.circle_parabola import CircleParabolaExpose
    
            solve_multiple_implicit = self.create_implicit_operation(
                CircleParabolaExpose())
            solve_multiple_implicit.declare_state('x', residual='rx')
            solve_multiple_implicit.declare_state('y', residual='ry')
            solve_multiple_implicit.linear_solver = ScipyKrylov()
            solve_multiple_implicit.nonlinear_solver = NewtonSolver(
                solve_subsystems=False)
    
            r = self.declare_variable('r', val=2)
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-3)
            c = self.declare_variable('c', val=2)
            x, y, t1, t2, t3, t4 = solve_multiple_implicit(
                r,
                a,
                b,
                c,
                expose=['t1', 't2', 't3', 't4'],
            )
    
    
    rep = GraphRepresentation(ExampleMultipleResidualsWithExposeDefineModelInline())
    sim = Simulator(rep)
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('y', sim['y'].shape)
    print(sim['y'])
    print('t1', sim['t1'].shape)
    print(sim['t1'])
    print('t2', sim['t2'].shape)
    print(sim['t2'])
    print('t3', sim['t3'].shape)
    print(sim['t3'])
    print('t4', sim['t4'].shape)
    print(sim['t4'])
    
    return sim, rep