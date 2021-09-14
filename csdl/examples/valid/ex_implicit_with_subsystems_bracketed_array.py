def example(Simulator):
    from csdl import Model, ImplicitModel, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleWithSubsystemsBracketedArray(ImplicitModel):
        def define(self):
            # define a subsystem (this is a very simple example)
            model = Model()
            p = model.create_input('p', val=[7, -7])
            q = model.create_input('q', val=[8, -8])
            r = p + q
            model.register_output('r', r)
    
            # add child system
            self.add(model, name='R')
            # declare output of child system as input to parent system
            r = self.declare_variable('r', shape=(2, ))
    
            c = self.declare_variable('c', val=[18, -18])
    
            # a == (3 + a - 2 * a**2)**(1 / 4)
            with self.create_model('coeff_a') as model:
                a = model.create_output('a')
                a.define((3 + a - 2 * a**2)**(1 / 4))
                model.nonlinear_solver = NonlinearBlockGS(iprint=0, maxiter=100)
    
            # store positive and negative values of `a` in an array
            ap = self.declare_variable('a')
            an = -ap
            a = self.create_output('vec_a', shape=(2, ))
            a[0] = ap
            a[1] = an
    
            with self.create_model('coeff_b') as model:
                model.create_input('b', val=[-4, 4])
    
            b = self.declare_variable('b', shape=(2, ))
            y = self.create_implicit_output('y', shape=(2, ))
            z = a * y**2 + b * y + c - r
            y.define_residual_bracketed(
                z,
                x1=[0, 2.],
                x2=[2, np.pi],
            )
    
    
    sim = Simulator(ExampleWithSubsystemsBracketedArray())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim