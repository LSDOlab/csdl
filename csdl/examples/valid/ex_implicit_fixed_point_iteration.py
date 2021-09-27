def example(Simulator):
    from csdl import Model, NonlinearBlockGS, ScipyKrylov, NewtonSolver
    import csdl
    import numpy as np

    class ExampleCycles(Model):
        def define(self):
            # x == (3 + x - 2 * x**2)**(1 / 4)
            m1 = Model()
            x = m1.declare_variable('a')
            r = m1.register_output('r', x - (3 + x - 2 * x**2)**(1 / 4))
            m1.print_var(r)
            a = self.implicit_operation(
                states=['a'],
                residuals=['r'],
                model=m1,
                nonlinear_solver=NewtonSolver(solve_subsystems=False),
                # nonlinear_solver=NonlinearBlockGS(maxiter=100),
                linear_solver=ScipyKrylov(),
            )

            # x == ((x + 3 - x**4) / 2)**(1 / 4)
            m2 = Model()
            x = m2.declare_variable('b')
            r = m2.register_output('r',
                                   x - ((x + 3 - x**4) / 2)**(1 / 4))
            m2.print_var(r)
            b = self.implicit_operation(
                states=['b'],
                residuals=['r'],
                model=m2,
                nonlinear_solver=NewtonSolver(solve_subsystems=False),
                # nonlinear_solver=NonlinearBlockGS(maxiter=100),
                linear_solver=ScipyKrylov(),
            )

            # x == 0.5 * x
            m3 = Model()
            x = m3.declare_variable('c')
            r = m3.register_output('r', x - 0.5 * x)
            m3.print_var(r)
            c = self.implicit_operation(
                states=['c'],
                residuals=['r'],
                model=m3,
                nonlinear_solver=NewtonSolver(solve_subsystems=False),
                # nonlinear_solver=NonlinearBlockGS(maxiter=100),
                linear_solver=ScipyKrylov(),
            )

    sim = Simulator(ExampleCycles())
    sim.run()

    print('a', sim['a'].shape)
    print(sim['a'])
    print('b', sim['b'].shape)
    print(sim['b'])
    print('c', sim['c'].shape)
    print(sim['c'])

    return sim


from csdl_om import Simulator

sim = example(Simulator)
# sim.visualize_implementation(recursive=True)
