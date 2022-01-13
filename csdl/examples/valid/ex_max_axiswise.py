def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleAxiswise(Model):
    
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
    
            # Computing the axiswise maximum on the tensor
            axis = 1
            ma = self.register_output('AxiswiseMax', csdl.max(ten,
                                                              axis=axis))
            assert ma.shape == (m, o, p, q)
    
    
    sim = Simulator(ExampleAxiswise())
    sim.run()
    
    print('tensor', sim['tensor'].shape)
    print(sim['tensor'])
    print('AxiswiseMax', sim['AxiswiseMax'].shape)
    print(sim['AxiswiseMax'])
    
    return sim