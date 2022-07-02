def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleWithSubsystemsBracketedScalarDefineModelInlineVariableBrackets(
            Model):
    
        def define(self):
            with self.create_submodel('R') as model:
                p = model.create_input('p', val=7)
                q = model.create_input('q', val=8)
                r = p + q
                model.register_output('r', r)
    
            m2 = Model()
            x = m2.declare_variable('a')
            r = m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))
    
            m3 = Model()
            a = m3.declare_variable('a')
            b = m3.declare_variable('b')
            c = m3.declare_variable('c')
            r = m3.declare_variable('r')
            y = m3.declare_variable('y')
            m3.register_output('z', a * y**2 + b * y + c - r)
    
            solve_fixed_point_iteration = self.create_implicit_operation(m2)
            solve_fixed_point_iteration.declare_state('a', residual='r')
            solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
    
            l = self.declare_variable('l', val=0)
            u = self.declare_variable('u', val=2)
    
            solve_quadratic = self.create_implicit_operation(m3)
            solve_quadratic.declare_state('y', residual='z', bracket=(l, u))
            solve_quadratic.nonlinear_solver = NonlinearBlockGS(maxiter=100)
    
            a = solve_fixed_point_iteration()
    
            b = self.declare_variable('b', val=-4)
            c = self.declare_variable('c', val=18)
            r = self.declare_variable('r')
            y = solve_quadratic(a, b, c, r)
    
    
    rep = GraphRepresentation(ExampleWithSubsystemsBracketedScalarDefineModelInlineVariableBrackets())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep