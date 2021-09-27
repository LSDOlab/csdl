def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np

    class ExampleCompositeResidual(Model):
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

            r = self.declare_variable('r', val=2)
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-3)
            c = self.declare_variable('c', val=2)
            x, y = self.implicit_operation(
                r,
                a,
                b,
                c,
                model=m,
                states=['x', 'y'],
                residuals=['rx', 'ry'],
                linear_solver=ScipyKrylov(),
                nonlinear_solver=NewtonSolver(solve_subsystems=False),
            )

    sim = Simulator(ExampleCompositeResidual())
    sim.run()

    print('x', sim['x'].shape)
    print(sim['x'])
    print('y', sim['y'].shape)
    print(sim['y'])

    return sim
