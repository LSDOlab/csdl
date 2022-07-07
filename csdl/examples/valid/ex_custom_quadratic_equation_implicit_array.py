def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleQuadraticEquationImplicitArray(Model):
    
        def define(self):
            from csdl.examples.operations.solve_quadratic import SolveQuadratic
    
            # These values overwrite the values within the CustomOperation
            a = self.declare_variable('a', val=[1, -1])
            b = self.declare_variable('b', val=[-4, 4])
            c = self.declare_variable('c', val=[3, -3])
    
            # Solve quadratic equation using a CustomImplicitOperation
            x = csdl.custom(a, b, c, op=SolveQuadratic())
            self.register_output('x', x)
    
    
    rep = GraphRepresentation(ExampleQuadraticEquationImplicitArray())
    sim = Simulator(rep)
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    
    return sim, rep