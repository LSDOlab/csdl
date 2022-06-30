def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ExampleImplicit2(Model):
    
        def initialize(self):
            self.parameters.declare('nlsolver')
    
        def define(self):
            # define internal model that defines a residual
            from csdl import NewtonSolver, ScipyKrylov
    
            solver_type = self.parameters['nlsolver']
    
            quadratic = Model()
            a = quadratic.declare_variable('a')
            b = quadratic.declare_variable('b')
            c = quadratic.declare_variable('c')
            x = quadratic.declare_variable('x')
            u = quadratic.declare_variable('u')
    
            # test_var = x**2
            # quadratic.register_output('test_var', test_var*2.0)
            # temp = quadratic.declare_variable('temp')
    
            # quadratic.connect(test_var.name, 'temp')
            # ax2 = a*temp
            # quadratic.register_output('t', a*1.0)
            ax2 = a * x**2
            au2 = a * u**2
    
            y = x - (-ax2 - c) / b
            v = u - (-au2 - c * 2) / b
    
            quadratic.register_output('y', y)
            quadratic.register_output('v', v)
    
            # from csdl_om import Simulator
            # sim = Simulator(quadratic)
            # sim.visualize_implementation()
            # exit()
    
            # SOLUTION: x [0.38742589]
            # SOLUTION: u [0.66666667]
    
            solve_quadratic = self.create_implicit_operation(quadratic)
            if solver_type == 'bracket':
                solve_quadratic.declare_state('x',
                                              residual='y',
                                              val=0.34,
                                              bracket=(0, 0.5))
                solve_quadratic.declare_state('u',
                                              residual='v',
                                              val=0.4,
                                              bracket=(0, 1.0))
            else:
                solve_quadratic.declare_state('x', residual='y', val=0.34)
                solve_quadratic.declare_state('u', residual='v', val=0.4)
                if solver_type == 'newton':
                    solve_quadratic.nonlinear_solver = NewtonSolver(
                        solve_subsystems=False)
                else:
                    raise ValueError(
                        f'solver type {solver_type} is unknown.')
    
            # solve_quadratic.linear_solver = csdl.LinearBlockGS()
            solve_quadratic.linear_solver = ScipyKrylov()
    
            aa = self.create_input('a', val=1.5)
            bb = self.create_input('b', val=2.0)
            cc = self.create_input('c', val=-1.0)
            xx, uu = solve_quadratic(aa, bb, cc)
    
            self.register_output('f', xx * 3.0 + uu * 3.0 + 0.5 * aa)
    
    
    rep = GraphRepresentation(ExampleImplicit2())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep