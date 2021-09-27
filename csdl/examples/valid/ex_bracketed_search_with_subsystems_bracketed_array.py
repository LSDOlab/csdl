def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleWithSubsystemsBracketedArray(Model):
        def define(self):
            with self.create_submodel('R') as model:
                p = model.create_input('p', val=[7, -7])
                q = model.create_input('q', val=[8, -8])
                r = p + q
                model.register_output('r', r)
    
            m2 = Model()
            x = m2.declare_variable('ap')
            r = m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))
    
            m3 = Model()
            a = m3.declare_variable('a', shape=(2, ))
            b = m3.declare_variable('b', shape=(2, ))
            c = m3.declare_variable('c', shape=(2, ))
            r = m3.declare_variable('r', shape=(2, ))
            y = m3.declare_variable('y', shape=(2, ))
            m3.register_output('z', a * y**2 + b * y + c - r)
    
            ap = self.implicit_operation(
                states=['ap'],
                residuals=['r'],
                model=m2,
                nonlinear_solver=NewtonSolver(solve_subsystems=False),
                # nonlinear_solver=NonlinearBlockGS(maxiter=100),
                linear_solver=ScipyKrylov(),
            )
    
            a = self.create_output('a', shape=(2, ))
            a[0] = ap
            a[1] = -ap
    
            b = self.declare_variable('b', val=[-4, 4])
            c = self.declare_variable('c', val=[18, -18])
            r = self.declare_variable('r', shape=(2, ))
            y = self.bracketed_search(
                a,
                b,
                c,
                r,
                model=m3,
                states=['y'],
                residuals=['z'],
                brackets=dict(y=(
                    np.array([0, 2.]),
                    np.array([2, np.pi], ),
                )),
            )
    
    
    sim = Simulator(ExampleWithSubsystemsBracketedArray())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim