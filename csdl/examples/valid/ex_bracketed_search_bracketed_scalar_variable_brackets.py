def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleBracketedScalarVariableBrackets(Model):
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.quadratic_function import QuadraticFunction
    
            l = self.declare_variable('l', val=0)
            u = self.declare_variable('u', val=2)
    
            solve_quadratic = self.create_implicit_operation(
                QuadraticFunction(shape=(1, )))
            solve_quadratic.declare_state('x', residual='y', bracket=(l, u))
    
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-4)
            c = self.declare_variable('c', val=3)
            x = solve_quadratic(a, b, c)
    
    
    rep = GraphRepresentation(ExampleBracketedScalarVariableBrackets())
    sim = Simulator(rep)
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    
    return sim, rep