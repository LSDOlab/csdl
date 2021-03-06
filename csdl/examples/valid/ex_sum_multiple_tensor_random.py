def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleMultipleTensorRandom(Model):
    
        def define(self):
            n = 3
            m = 6
            p = 7
            q = 10
            np.random.seed(0)
    
            # Declare a tensor of shape 3x6x7x10 as input
            T1 = self.declare_variable(
                'T1',
                val=np.random.rand(n * m * p * q).reshape((n, m, p, q)))
    
            # Declare another tensor of shape 3x6x7x10 as input
            T2 = self.declare_variable(
                'T2',
                val=np.random.rand(n * m * p * q).reshape((n, m, p, q)))
    
            # Output the elementwise sum of tensors T1 and T2
            self.register_output('multiple_tensor_sum', csdl.sum(T1, T2))
    
    
    rep = GraphRepresentation(ExampleMultipleTensorRandom())
    sim = Simulator(rep)
    sim.run()
    
    print('T1', sim['T1'].shape)
    print(sim['T1'])
    print('T2', sim['T2'].shape)
    print(sim['T2'])
    print('multiple_tensor_sum', sim['multiple_tensor_sum'].shape)
    print(sim['multiple_tensor_sum'])
    
    return sim, rep