def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleTensor2Vector(Model):
        def define(self):
            shape = (2, 3, 4, 5)
            size = 2 * 3 * 4 * 5
    
            # Both vector or tensors need to be numpy arrays
            tensor = np.arange(size).reshape(shape)
            vector = np.arange(size)
    
            # in1 is a tensor having shape = (2, 3, 4, 5)
            in1 = self.declare_variable('in1', val=tensor)
    
            # in1 is being reshaped from shape = (2, 3, 4, 5) to a vector
            # having size = 2 * 3 * 4 * 5
            self.register_output('reshape_tensor2vector',
                                 csdl.reshape(in1, new_shape=(size, )))
    
    
    rep = GraphRepresentation(ExampleTensor2Vector())
    sim = Simulator(rep)
    sim.run()
    
    print('in1', sim['in1'].shape)
    print(sim['in1'])
    print('reshape_tensor2vector', sim['reshape_tensor2vector'].shape)
    print(sim['reshape_tensor2vector'])
    
    return sim, rep