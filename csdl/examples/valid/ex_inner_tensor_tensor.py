def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleTensorTensor(Model):
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
    
            # Adding tensors to csdl
            ten1 = self.declare_variable('ten1', val=ten1)
            ten2 = self.declare_variable('ten2', val=ten2)
    
            # Tensor-Tensor Inner Product specifying the first and last axes
            self.register_output(
                'TenTenInner',
                csdl.inner(ten1, ten2, axes=([0, 2], [0, 2])),
            )
    
    
    sim = Simulator(ExampleTensorTensor())
    sim.run()
    
    print('ten1', sim['ten1'].shape)
    print(sim['ten1'])
    print('ten2', sim['ten2'].shape)
    print(sim['ten2'])
    print('TenTenInner', sim['TenTenInner'].shape)
    print(sim['TenTenInner'])
    
    return sim