def example(Simulator):
    from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np

    class ExampleApplyNonlinear(Model):
        def define(self):
            # define internal model that defines a residual
            model = Model()
            a = model.declare_variable('a', val=1)
            b = model.declare_variable('b', val=-4)
            c = model.declare_variable('c', val=3)
            x = model.declare_variable('x')
            y = a * x**2 + b * x + c
            model.register_output('y', y)

            # define arguments to implicit operation
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-4)
            c = self.declare_variable('c', val=3)

            # define output of implicit operation
            x = self.implicit_operation(
                a,
                b,
                c,
                states=['x'],
                residuals=['y'],
                model=model,
                nonlinear_solver=NewtonSolver(solve_subsystems=False),
                linear_solver=ScipyKrylov(),
            )

    sim = Simulator(ExampleApplyNonlinear())
    sim.run()

    return sim


from csdl_om import Simulator

sim = example(Simulator)
# sim.visualize_implementation(recursive=True)
sim['x'] = 1.9
sim.run()
print(sim['x'])
sim['x'] = 2.1
sim.run()
print(sim['x'])
