def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleScalar(Model):
    
        def define(self):
            m = 2
            n = 3
            o = 4
            p = 5
            q = 6
    
            # Shape of a tensor
            tensor_shape = (m, n, o, p, q)
    
            num_of_elements = np.prod(tensor_shape)
            # Creating the values of the tensor
            val = np.arange(num_of_elements).reshape(tensor_shape)
    
            # Declaring the tensor as an input
            ten = self.declare_variable('tensor', val=val)
    
            # Computing the maximum across the entire tensor, returns single value
            ma = self.register_output('ScalarMax', csdl.max(ten))
            assert ma.shape == (1, ), ma.shape
    
    
    rep = GraphRepresentation(ExampleScalar())
    sim = Simulator(rep)
    sim.run()
    
    print('tensor', sim['tensor'].shape)
    print(sim['tensor'])
    print('ScalarMax', sim['ScalarMax'].shape)
    print(sim['ScalarMax'])
    
    return sim, rep