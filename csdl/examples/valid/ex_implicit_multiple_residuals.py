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
    
    
    class ExampleMultipleResiduals(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.circle_parabola import CircleParabola
            r = self.declare_variable('r', val=2)
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-3)
            c = self.declare_variable('c', val=2)
            solve_multiple_implicit = self.create_implicit_operation(
                CircleParabola())
            solve_multiple_implicit.declare_state('x',
                                                  residual='rx',
                                                  val=1.5)
            solve_multiple_implicit.declare_state('y',
                                                  residual='ry',
                                                  val=0.9)
            solve_multiple_implicit.linear_solver = ScipyKrylov()
            solve_multiple_implicit.nonlinear_solver = NewtonSolver(
                solve_subsystems=False)
    
            x, y = solve_multiple_implicit(r, a, b, c)
    
    
    rep = GraphRepresentation(ExampleMultipleResiduals())
    sim = Simulator(rep)
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep