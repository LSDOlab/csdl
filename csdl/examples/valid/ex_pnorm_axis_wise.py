def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleAxisWise(Model):
        def define(self):
    
            # Shape of the tensor
            shape = (2, 3, 4, 5)
    
            # Number of elements in the tensor
            num_of_elements = np.prod(shape)
    
            # Creating a numpy tensor with the desired shape and size
            tensor = np.arange(num_of_elements).reshape(shape)
    
            # Declaring in1 as input tensor
            in1 = self.declare_variable('in1', val=tensor)
    
            # Computing the 6-norm of in1 over the specified axes.
            self.register_output('axiswise_pnorm',
                                 csdl.pnorm(in1, axis=(1, 3), pnorm_type=6))
    
    
    sim = Simulator(ExampleAxisWise())
    sim.run()
    
    print('in1', sim['in1'].shape)
    print(sim['in1'])
    print('axiswise_pnorm', sim['axiswise_pnorm'].shape)
    print(sim['axiswise_pnorm'])
    
    return sim