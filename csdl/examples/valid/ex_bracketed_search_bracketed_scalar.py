def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleBracketedScalar(Model):
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.quadratic_function import QuadraticFunction
    
            solve_quadratic = self.create_implicit_operation(
                QuadraticFunction(shape=(1, )))
            solve_quadratic.declare_state('x', residual='y', bracket=(0, 2))
    
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-4)
            c = self.declare_variable('c', val=3)
            x = solve_quadratic(a, b, c)
    
    
    rep = GraphRepresentation(ExampleBracketedScalar())
    sim = Simulator(rep)
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    
    return sim, rep