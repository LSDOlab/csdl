def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleMultipleVectorRandom(Model):
    
        def define(self):
            n = 3
            np.random.seed(0)
    
            # Declare a vector of length 3 as input
            v1 = self.declare_variable('v1', val=np.random.rand(n))
    
            # Declare another vector of length 3 as input
            v2 = self.declare_variable('v2', val=np.random.rand(n))
    
            # Output the elementwise sum of vectors v1 and v2
            self.register_output('multiple_vector_sum', csdl.sum(v1, v2))
    
    
    rep = GraphRepresentation(ExampleMultipleVectorRandom())
    sim = Simulator(rep)
    sim.run()
    
    print('v1', sim['v1'].shape)
    print(sim['v1'])
    print('v2', sim['v2'].shape)
    print(sim['v2'])
    print('multiple_vector_sum', sim['multiple_vector_sum'].shape)
    print(sim['multiple_vector_sum'])
    
    return sim, rep