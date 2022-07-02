def example(Simulator):
    from csdl import Model, GraphRepresentation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleWithSubsystemsBracketedScalarVariableBrackets(Model):
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
            from csdl.examples.models.simple_add import SimpleAdd
            from csdl.examples.models.fixed_point import FixedPoint2
            self.add(SimpleAdd(p=7, q=8), name='R')
            solve_fixed_point_iteration = self.create_implicit_operation(
                FixedPoint2(name='a'))
            solve_fixed_point_iteration.declare_state('a', residual='r')
            solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
    
            l = self.declare_variable('l', val=0)
            u = self.declare_variable('u', val=2)
    
            solve_quadratic = self.create_implicit_operation(
                QuadraticWithExtraTerm(shape=(1, )))
            solve_quadratic.declare_state('y', residual='z', bracket=(l, u))
            solve_quadratic.nonlinear_solver = NonlinearBlockGS(maxiter=100)
    
            a = solve_fixed_point_iteration()
    
            b = self.declare_variable('b', val=-4)
            c = self.declare_variable('c', val=18)
            r = self.declare_variable('r')
            y = solve_quadratic(a, b, c, r)
    
    
    rep = GraphRepresentation(ExampleWithSubsystemsBracketedScalarVariableBrackets())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep