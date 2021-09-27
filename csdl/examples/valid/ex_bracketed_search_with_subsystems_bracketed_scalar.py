def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleWithSubsystemsBracketedScalar(Model):
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
    
            a = self.implicit_operation(
                states=['a'],
                residuals=['r'],
                model=m2,
                nonlinear_solver=NewtonSolver(solve_subsystems=False),
                # nonlinear_solver=NonlinearBlockGS(maxiter=100),
                linear_solver=ScipyKrylov(),
            )
    
            b = self.declare_variable('b', val=-4)
            c = self.declare_variable('c', val=18)
            r = self.declare_variable('r')
            y = self.bracketed_search(
                a,
                b,
                c,
                r,
                model=m3,
                states=['y'],
                residuals=['z'],
                brackets=dict(y=(0, 2)),
            )
    
    
    sim = Simulator(ExampleWithSubsystemsBracketedScalar())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim