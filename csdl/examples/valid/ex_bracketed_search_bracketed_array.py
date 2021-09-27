def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleBracketedArray(Model):
        def define(self):
            model = Model()
            a = model.declare_variable('a', shape=(2, ))
            b = model.declare_variable('b', shape=(2, ))
            c = model.declare_variable('c', shape=(2, ))
            x = model.declare_variable('x', shape=(2, ))
            y = a * x**2 + b * x + c
            model.register_output('y', y)
    
            a = model.declare_variable('a', val=[1, -1])
            b = model.declare_variable('b', val=[-4, 4])
            c = model.declare_variable('c', val=[3, -3])
            x = self.bracketed_search(
                a,
                b,
                c,
                states=['x'],
                residuals=['y'],
                model=model,
                brackets=dict(x=(
                    np.array([0, 2.]),
                    np.array([2, np.pi], ),
                )),
            )
    
    
    sim = Simulator(ExampleBracketedArray())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    
    return sim