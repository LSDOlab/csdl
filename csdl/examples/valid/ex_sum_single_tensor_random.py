def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleSingleTensorRandom(Model):
    
        def define(self):
            n = 3
            m = 4
            p = 5
            q = 6
            np.random.seed(0)
    
            # Declare a tensor of shape 3x6x7x10 as input
            T1 = self.declare_variable(
                'T1',
                val=np.random.rand(n * m * p * q).reshape((n, m, p, q)))
            # Output the sum of all the elements of the matrix M1
            self.register_output('single_tensor_sum', csdl.sum(T1))
    
    
    rep = GraphRepresentation(ExampleSingleTensorRandom())
    sim = Simulator(rep)
    sim.run()
    
    print('T1', sim['T1'].shape)
    print(sim['T1'])
    print('single_tensor_sum', sim['single_tensor_sum'].shape)
    print(sim['single_tensor_sum'])
    
    return sim, rep