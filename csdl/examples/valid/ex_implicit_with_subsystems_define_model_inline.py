def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleWithSubsystemsDefineModelInline(Model):
        def define(self):
            with self.create_submodel('R') as model:
                p = model.create_input('p', val=7)
                q = model.create_input('q', val=8)
                r = p + q
                model.register_output('r', r)
            r = self.declare_variable('r')
    
            m2 = Model()
            a = m2.declare_variable('a')
            r = m2.register_output('r', a - ((a + 3 - a**4) / 2)**(1 / 4))
    
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
            a = solve_fixed_point_iteration()
    
            solve_quadratic = self.create_implicit_operation(m3)
            b = self.create_input('b', val=-4)
            solve_quadratic.declare_state('y', residual='z')
            solve_quadratic.nonlinear_solver = NewtonSolver(
                solve_subsystems=False,
                maxiter=100,
                iprint=False,
            )
            solve_quadratic.linear_solver = ScipyKrylov()
    
            c = self.declare_variable('c', val=18)
            y = solve_quadratic(a, b, c, r)
    
    
    rep = GraphRepresentation(ExampleWithSubsystemsDefineModelInline())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep