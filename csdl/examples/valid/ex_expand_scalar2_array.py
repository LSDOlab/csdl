def example(Simulator):
    import csdl
    from csdl import Model
    import numpy as np
    
    
    class ExampleScalar2Array(Model):
        def define(self):
            # Expanding a scalar into an array
            scalar = self.declare_variable('scalar', val=1.)
            expanded_scalar = csdl.expand(scalar, (2, 3))
            self.register_output('expanded_scalar', expanded_scalar)
    
    
    sim = Simulator(ExampleScalar2Array())
    sim.run()
    
    print('scalar', sim['scalar'].shape)
    print(sim['scalar'])
    print('expanded_scalar', sim['expanded_scalar'].shape)
    print(sim['expanded_scalar'])
    
    return sim