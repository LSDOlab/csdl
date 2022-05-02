def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleVector2Tensor(Model):
        def define(self):
            shape = (2, 3, 4, 5)
            size = 2 * 3 * 4 * 5
    
            # Both vector or tensors need to be numpy arrays
            tensor = np.arange(size).reshape(shape)
            vector = np.arange(size)
    
            # in2 is a vector having a size of 2 * 3 * 4 * 5
            in2 = self.declare_variable('in2', val=vector)
    
            # in2 is being reshaped from size =  2 * 3 * 4 * 5 to a tenßsor
            # having shape = (2, 3, 4, 5)
            self.register_output('reshape_vector2tensor',
                                 csdl.reshape(in2, new_shape=shape))
    
    
    rep = GraphRepresentation(ExampleVector2Tensor())
    sim = Simulator(rep)
    sim.run()
    
    print('in2', sim['in2'].shape)
    print(sim['in2'])
    print('reshape_vector2tensor', sim['reshape_vector2tensor'].shape)
    print(sim['reshape_vector2tensor'])
    
    return sim, rep