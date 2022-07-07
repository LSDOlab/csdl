def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleConcatenateRandom(Model):
    
        def define(self):
            n = 5
            np.random.seed(0)
    
            # Declare a vector of length 3 as input
            v1 = self.declare_variable('v1', val=np.random.rand(n))
            v2 = self.declare_variable('v2', val=np.random.rand(n - 1))
            v3 = self.declare_variable('v3', val=np.zeros(n))
    
            # Output the sum of all the elements of the vector v1
            single_vector_sum_1a = csdl.sum(v1, axes=(0, ))
            single_vector_sum_1b = csdl.sum(v1)
            self.register_output('single_vector_sum_1a',
                                 single_vector_sum_1a)
            self.register_output('single_vector_sum_1b',
                                 single_vector_sum_1b)
            single_vector_sum_2 = self.register_output(
                'single_vector_sum_2', csdl.sum(v2, axes=(0, )))
            single_vector_sum_3 = csdl.sum(v3)
            self.register_output('single_vector_sum_3', single_vector_sum_3)
    
            sum_vector = self.create_output(name='sum_vector', shape=(3, ))
    
            sum_vector[0] = single_vector_sum_1a
            sum_vector[1] = single_vector_sum_2
            sum_vector[2] = single_vector_sum_3
    
    
    rep = GraphRepresentation(ExampleConcatenateRandom())
    sim = Simulator(rep)
    sim.run()
    
    print('single_vector_sum_1a', sim['single_vector_sum_1a'].shape)
    print(sim['single_vector_sum_1a'])
    print('single_vector_sum_1b', sim['single_vector_sum_1b'].shape)
    print(sim['single_vector_sum_1b'])
    print('single_vector_sum_2', sim['single_vector_sum_2'].shape)
    print(sim['single_vector_sum_2'])
    print('single_vector_sum_3', sim['single_vector_sum_3'].shape)
    print(sim['single_vector_sum_3'])
    print('sum_vector', sim['sum_vector'].shape)
    print(sim['sum_vector'])
    
    return sim, rep