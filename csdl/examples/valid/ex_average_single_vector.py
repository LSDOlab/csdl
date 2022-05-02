def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleSingleVector(Model):
        def define(self):
            n = 3
    
            # Declare a vector of length 3 as input
            v1 = self.declare_variable('v1', val=np.arange(n))
    
            # Output the average of all the elements of the vector v1
            self.register_output('single_vector_average', csdl.average(v1))
    
    
    rep = GraphRepresentation(ExampleSingleVector())
    sim = Simulator(rep)
    sim.run()
    
    print('v1', sim['v1'].shape)
    print(sim['v1'])
    print('single_vector_average', sim['single_vector_average'].shape)
    print(sim['single_vector_average'])
    
    return sim, rep