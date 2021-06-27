def example(Simulator):
    from csdl import Model, ImplicitModel, ScipyKrylov, NewtonSolver, NonlinearBlockGS
    import numpy as np
    
    
    class ExampleCompositeResidual(ImplicitModel):
        def define(self):
            r = self.declare_variable('r', val=2)
            a = self.declare_variable('a', val=1)
            b = self.declare_variable('b', val=-3)
            c = self.declare_variable('c', val=2)
            x = self.create_implicit_output('x', val=1.5)
            y = self.create_implicit_output('y', val=0.9)
    
            x.define_residual(x**2 + (y - r)**2 - r**2)
            y.define_residual(a * y**2 + b * y + c)
            self.linear_solver = ScipyKrylov()
            self.nonlinear_solver = NewtonSolver(solve_subsystems=False)
    
    
    sim = Simulator(ExampleCompositeResidual())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim