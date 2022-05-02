def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    from csdl.examples.models.quadratic_function import QuadraticFunction
    from csdl.examples.models.quadratic_function import QuadraticFunction
    from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
    from csdl.examples.models.simple_add import SimpleAdd
    from csdl.examples.models.fixed_point import FixedPoint2
    from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
    from csdl.examples.models.simple_add import SimpleAdd
    from csdl.examples.models.fixed_point import FixedPoint2
    
    
    class ExampleWithSubsystemsBracketedArray(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
            from csdl.examples.models.simple_add import SimpleAdd
            from csdl.examples.models.fixed_point import FixedPoint2
            self.add(SimpleAdd(p=[7, -7], q=[8, -8]), name='R')
    
            solve_fixed_point_iteration = self.create_implicit_operation(
                FixedPoint2(name='ap'))
            solve_fixed_point_iteration.declare_state('ap', residual='r')
            solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
    
            solve_quadratic = self.create_implicit_operation(
                QuadraticWithExtraTerm(shape=(2, )))
            solve_quadratic.declare_state('y',
                                          residual='z',
                                          bracket=(
                                              np.array([0, 2.]),
                                              np.array([2, np.pi], ),
                                          ))
            solve_quadratic.nonlinear_solver = NonlinearBlockGS(maxiter=100)
    
            ap = solve_fixed_point_iteration()
            a = self.create_output('a', shape=(2, ))
            a[0] = ap
            a[1] = -ap
    
            b = self.declare_variable('b', val=[-4, 4])
            c = self.declare_variable('c', val=[18, -18])
            r = self.declare_variable('r', shape=(2, ))
            y = solve_quadratic(a, b, c, r)
    
    
    rep = GraphRepresentation(ExampleWithSubsystemsBracketedArray())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep