def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleWithSubsystems(Model):
        def define(self):
            with self.create_submodel('R') as model:
                p = model.create_input('p', val=7)
                q = model.create_input('q', val=8)
                r = p + q
                model.register_output('r', r)
            r = self.declare_variable('r')
    
            m2 = Model()
            a = m2.declare_variable('a')
            m2.register_output('r', a - (3 + a - 2 * a**2)**(1 / 4))
    
            # x == ((x + 3 - x**4) / 2)**(1 / 4)
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
            m3.print_var(y)
    
            a = self.implicit_operation(
                states=['a'],
                residuals=['r'],
                model=m2,
                nonlinear_solver=NewtonSolver(
                    solve_subsystems=False,
                    maxiter=100,
                    iprint=False,
                ),
                # nonlinear_solver=NonlinearBlockGS(maxiter=100),
                linear_solver=ScipyKrylov(),
            )
    
            b = self.create_input('b', val=-4)
            c = self.declare_variable('c', val=18)
            y = self.implicit_operation(
                a,
                b,
                c,
                r,
                states=['y'],
                residuals=['z'],
                model=m3,
                nonlinear_solver=NewtonSolver(
                    solve_subsystems=False,
                    maxiter=100,
                    iprint=False,
                ),
                linear_solver=ScipyKrylov(),
            )
    
    
    sim = Simulator(ExampleWithSubsystems())
    sim.run()
    
    return sim