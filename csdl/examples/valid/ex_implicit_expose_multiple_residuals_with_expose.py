def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleMultipleResidualsWithExpose(Model):
        def define(self):
            m = Model()
            r = m.declare_variable('r')
            a = m.declare_variable('a')
            b = m.declare_variable('b')
            c = m.declare_variable('c')
            x = m.declare_variable('x', val=1.5)
            y = m.declare_variable('y', val=0.9)
            m.register_output('rx', x**2 + (y - r)**2 - r**2)
            m.register_output('ry', a * y**2 + b * y + c)
            m.register_output('t1', a + b + c)
            m.register_output('t2', x**2)
            m.register_output('t3', 2 * y)
            m.register_output('t4', x + y)
    
            r = self.declare_variable('r', val=2)
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-3)
            c = self.declare_variable('c', val=2)
            solve_multiple_implicit = self.create_implicit_operation(m)
            solve_multiple_implicit.declare_state('x', residual='rx')
            solve_multiple_implicit.declare_state('y', residual='ry')
            solve_multiple_implicit.linear_solver = ScipyKrylov()
            solve_multiple_implicit.nonlinear_solver = NewtonSolver(
                solve_subsystems=False)
    
            x, y, t1, t2, t3, t4 = solve_multiple_implicit(
                r,
                a,
                b,
                c,
                expose=['t1', 't2', 't3', 't4'],
            )
    
    
    sim = Simulator(ExampleMultipleResidualsWithExpose())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('y', sim['y'].shape)
    print(sim['y'])
    print('t1', sim['t1'].shape)
    print(sim['t1'])
    print('t2', sim['t2'].shape)
    print(sim['t2'])
    print('t3', sim['t3'].shape)
    print(sim['t3'])
    print('t4', sim['t4'].shape)
    print(sim['t4'])
    
    return sim