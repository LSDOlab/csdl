def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleBracketedScalar(Model):
        def define(self):
            with self.create_submodel('sys') as model:
                a = model.declare_variable('a', val=1)
                b = model.declare_variable('b', val=-4)
                c = model.declare_variable('c', val=3)
    
                x = model.declare_variable('x')
    
                y = a * x**2 + b * x + c
                model.register_output('y', y)
    
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-4)
            c = self.declare_variable('c', val=3)
            x = self.bracketed_search(
                a,
                b,
                c,
                states=['x'],
                residuals=['y'],
                model=model,
                brackets=dict(y=(0, 2)),
            )
    
    
    sim = Simulator(ExampleBracketedScalar())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    
    return sim