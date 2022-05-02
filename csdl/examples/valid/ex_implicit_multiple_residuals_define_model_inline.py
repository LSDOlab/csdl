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
    
    
    class ExampleMultipleResidualsDefineModelInline(Model):
        def define(self):
            m = Model()
            r = m.declare_variable('r')
            a = m.declare_variable('a')
            b = m.declare_variable('b')
            c = m.declare_variable('c')
            x = m.declare_variable('x', val=1.5)
            y = m.declare_variable('y', val=0.9)
            m.register_output('rx', x**2 + (y - r)**2 - r**2)
            m.register_output('ry', a * y**2 + b * y + c)
    
            r = self.declare_variable('r', val=2)
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-3)
            c = self.declare_variable('c', val=2)
            solve_multiple_implicit = self.create_implicit_operation(m)
            solve_multiple_implicit.declare_state('x', residual='rx')
            solve_multiple_implicit.declare_state('y', residual='ry')
            solve_multiple_implicit.linear_solver = ScipyKrylov()
            solve_multiple_implicit.nonlinear_solver = NewtonSolver(
                solve_subsystems=False)
    
            x, y = solve_multiple_implicit(r, a, b, c)
    
    
    rep = GraphRepresentation(ExampleMultipleResidualsDefineModelInline())
    sim = Simulator(rep)
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep