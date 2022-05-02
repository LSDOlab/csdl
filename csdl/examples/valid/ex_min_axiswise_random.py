def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleAxiswiseRandom(Model):
    
        def define(self):
            m = 2
            n = 3
            o = 4
            p = 5
            q = 6
            np.random.seed(0)
    
            # Shape of a tensor
            tensor_shape = (m, n, o, p, q)
    
            num_of_elements = np.prod(tensor_shape)
            # Creating the values of the tensor
            val = np.random.rand(num_of_elements).reshape(tensor_shape)
    
            # Declaring the tensor as an input
            ten = self.declare_variable('tensor', val=val)
    
            # Computing the axiswise minimum on the tensor
            axis = 1
            ma = self.register_output('AxiswiseMin',
                                      csdl.min(ten, axis=axis, rho=75000.))
            assert ma.shape == (m, o, p, q)
    
    
    rep = GraphRepresentation(ExampleAxiswiseRandom())
    sim = Simulator(rep)
    sim.run()
    
    print('tensor', sim['tensor'].shape)
    print(sim['tensor'])
    print('AxiswiseMin', sim['AxiswiseMin'].shape)
    print(sim['AxiswiseMin'])
    
    return sim, rep