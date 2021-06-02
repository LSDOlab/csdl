def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleSingleVector(Model):
        def define(self):
            n = 3
    
            # Declare a vector of length 3 as input
            v1 = self.declare_variable('v1', val=np.arange(n))
    
            # Output the sum of all the elements of the vector v1
            self.register_output('single_vector_sum', csdl.sum(v1))
    
    
    sim = Simulator(ExampleSingleVector())
    sim.run()
    
    print('v1', sim['v1'].shape)
    print(sim['v1'])
    print('single_vector_sum', sim['single_vector_sum'].shape)
    print(sim['single_vector_sum'])
    
    return sim