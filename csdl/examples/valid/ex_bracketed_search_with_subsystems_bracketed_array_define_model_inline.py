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
    
    
    class ExampleWithSubsystemsBracketedArrayDefineModelInline(Model):
        def define(self):
            with self.create_submodel('R') as model:
                p = model.create_input('p', val=[7, -7])
                q = model.create_input('q', val=[8, -8])
                r = p + q
                model.register_output('r', r)
    
            m2 = Model()
            x = m2.declare_variable('ap')
            r = m2.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))
    
            m3 = Model()
            a = m3.declare_variable('a', shape=(2, ))
            b = m3.declare_variable('b', shape=(2, ))
            c = m3.declare_variable('c', shape=(2, ))
            r = m3.declare_variable('r', shape=(2, ))
            y = m3.declare_variable('y', shape=(2, ))
            m3.register_output('z', a * y**2 + b * y + c - r)
    
            solve_fixed_point_iteration = self.create_implicit_operation(m2)
            solve_fixed_point_iteration.declare_state('ap', residual='r')
            solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(
                maxiter=100)
    
            solve_quadratic = self.create_implicit_operation(m3)
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
    
    
    rep = GraphRepresentation(ExampleWithSubsystemsBracketedArrayDefineModelInline())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep