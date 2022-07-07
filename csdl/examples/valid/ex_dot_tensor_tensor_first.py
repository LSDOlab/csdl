def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleTensorTensorFirst(Model):
        def define(self):
    
            m = 3
            n = 4
            p = 5
    
            # Shape of the tensors
            ten_shape = (m, n, p)
    
            # Number of elements in the tensors
            num_ten_elements = np.prod(ten_shape)
    
            # Values for the two tensors
            ten1 = np.arange(num_ten_elements).reshape(ten_shape)
            ten2 = np.arange(num_ten_elements,
                             2 * num_ten_elements).reshape(ten_shape)
    
            # Adding the tensors to csdl
            ten1 = self.declare_variable('ten1', val=ten1)
            ten2 = self.declare_variable('ten2', val=ten2)
    
            # Tensor-Tensor Dot Product specifying the first axis
            self.register_output('TenTenDotFirst', csdl.dot(ten1, ten2, axis=0))
    
    
    rep = GraphRepresentation(ExampleTensorTensorFirst())
    sim = Simulator(rep)
    sim.run()
    
    print('ten1', sim['ten1'].shape)
    print(sim['ten1'])
    print('ten2', sim['ten2'].shape)
    print(sim['ten2'])
    print('TenTenDotFirst', sim['TenTenDotFirst'].shape)
    print(sim['TenTenDotFirst'])
    
    return sim, rep