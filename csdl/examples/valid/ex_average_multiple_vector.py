def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleMultipleVector(Model):
        def define(self):
            n = 3
    
            # Declare a vector of length 3 as input
            v1 = self.declare_variable('v1', val=np.arange(n))
    
            # Declare another vector of length 3 as input
            v2 = self.declare_variable('v2', val=np.arange(n, 2 * n))
    
            # Output the elementwise average of vectors v1 and v2
            self.register_output('multiple_vector_average', csdl.average(v1, v2))
    
    
    sim = Simulator(ExampleMultipleVector())
    sim.run()
    
    print('v1', sim['v1'].shape)
    print(sim['v1'])
    print('v2', sim['v2'].shape)
    print(sim['v2'])
    print('multiple_vector_average', sim['multiple_vector_average'].shape)
    print(sim['multiple_vector_average'])
    
    return sim