def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleBracketedArrayDefineModelInlineVariableBrackets(Model):
    
        def define(self):
            model = Model()
            a = model.declare_variable('a', shape=(2, ))
            b = model.declare_variable('b', shape=(2, ))
            c = model.declare_variable('c', shape=(2, ))
            x = model.declare_variable('x', shape=(2, ))
            y = a * x**2 + b * x + c
            model.register_output('y', y)
    
            l = self.declare_variable('l', val=np.array([0, 2.]))
            u = self.declare_variable('u', val=np.array([2, np.pi]))
    
            solve_quadratic = self.create_implicit_operation(model)
            solve_quadratic.declare_state('x', residual='y', bracket=(l, u))
    
            a = self.declare_variable('a', val=[1, -1])
            b = self.declare_variable('b', val=[-4, 4])
            c = self.declare_variable('c', val=[3, -3])
            x = solve_quadratic(a, b, c)
    
    
    rep = GraphRepresentation(ExampleBracketedArrayDefineModelInlineVariableBrackets())
    sim = Simulator(rep)
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    
    return sim, rep