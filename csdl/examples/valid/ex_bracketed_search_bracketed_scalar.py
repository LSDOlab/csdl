def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleBracketedScalar(Model):
        def define(self):
            model = Model()
            a = model.declare_variable('a')
            b = model.declare_variable('b')
            c = model.declare_variable('c')
            x = model.declare_variable('x')
            y = a * x**2 + b * x + c
            model.register_output('y', y)
    
            solve_quadratic = self.create_implicit_operation(model)
            solve_quadratic.declare_state('x', residual='y', bracket=(0, 2))
    
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-4)
            c = self.declare_variable('c', val=3)
            x = solve_quadratic(a, b, c)
    
    
    sim = Simulator(ExampleBracketedScalar())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    
    return sim