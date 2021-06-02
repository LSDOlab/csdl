def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleTensorTensor(Model):
        def define(self):
            # Creating two tensors
            shape = (2, 5, 4, 3)
            num_elements = np.prod(shape)
    
            tenval1 = np.arange(num_elements).reshape(shape)
            tenval2 = np.arange(num_elements).reshape(shape) + 6
    
            ten1 = self.declare_variable('ten1', val=tenval1)
            ten2 = self.declare_variable('ten2', val=tenval2)
    
            # Tensor-Tensor Dot Product specifying the last axis
            self.register_output('TenTenCross', csdl.cross(ten1, ten2, axis=3))
    
    
    sim = Simulator(ExampleTensorTensor())
    sim.run()
    
    print('ten1', sim['ten1'].shape)
    print(sim['ten1'])
    print('ten2', sim['ten2'].shape)
    print(sim['ten2'])
    print('TenTenCross', sim['TenTenCross'].shape)
    print(sim['TenTenCross'])
    
    return sim