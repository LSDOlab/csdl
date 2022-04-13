def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    from csdl.examples.models.quadratic_function import QuadraticFunction
    from csdl.examples.models.fixed_point import FixedPoint1, FixedPoint2, FixedPoint3
    from csdl.examples.models.fixed_point import FixedPoint2
    from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
    from csdl.examples.models.simple_add import SimpleAdd
    from csdl.examples.models.circle_parabola import CircleParabola
    from csdl.examples.models.quadratic_function import QuadraticFunction
    
    
    class ExampleWithSubsystems(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.fixed_point import FixedPoint2
            from csdl.examples.models.quadratic_wih_extra_term import QuadraticWithExtraTerm
            from csdl.examples.models.simple_add import SimpleAdd
    
            self.add(SimpleAdd(p=7, q=8), name='R')
            r = self.declare_variable('r')
    
            solve_fixed_point_iteration = self.create_implicit_operation(
                FixedPoint2(name='a'))
            solve_fixed_point_iteration.declare_state('a', residual='r')
            solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
            a = solve_fixed_point_iteration()
    
            solve_quadratic = self.create_implicit_operation(
                QuadraticWithExtraTerm(shape=(1, )))
            b = self.create_input('b', val=-4)
            solve_quadratic.declare_state('y', residual='z')
            solve_quadratic.nonlinear_solver = NewtonSolver(
                solve_subsystems=False,
                maxiter=100,
                iprint=False,
            )
            solve_quadratic.linear_solver = ScipyKrylov()
    
            c = self.declare_variable('c', val=18)
            y = solve_quadratic(a, b, c, r)
    
    
    sim = Simulator(ExampleWithSubsystems())
    sim.run()
    
    return sim