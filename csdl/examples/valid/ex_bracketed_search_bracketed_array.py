def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    from csdl.examples.models.quadratic_function import QuadraticFunction
    from csdl.examples.models.quadratic_function import QuadraticFunction
    from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
    from csdl.examples.models.simple_add import SimpleAdd
    from csdl.examples.models.fixed_point import FixedPoint2
    from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
    from csdl.examples.models.simple_add import SimpleAdd
    from csdl.examples.models.fixed_point import FixedPoint2
    
    
    class ExampleBracketedArray(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.quadratic_function import QuadraticFunction
            solve_quadratic = self.create_implicit_operation(
                QuadraticFunction(shape=(2, )))
            solve_quadratic.declare_state('x',
                                          residual='y',
                                          bracket=(
                                              np.array([0, 2.]),
                                              np.array([2, np.pi], ),
                                          ))
    
            a = self.declare_variable('a', val=[1, -1])
            b = self.declare_variable('b', val=[-4, 4])
            c = self.declare_variable('c', val=[3, -3])
            x = solve_quadratic(a, b, c)
    
    
    sim = Simulator(ExampleBracketedArray())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    
    return sim