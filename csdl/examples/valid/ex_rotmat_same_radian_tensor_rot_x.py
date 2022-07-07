def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleSameRadianTensorRotX(Model):
        def define(self):
    
            # Shape of a random tensor rotation matrix
            shape = (2, 3, 4)
    
            num_elements = np.prod(shape)
    
            # Tensor of angles in radians
            angle_val1 = np.repeat(np.pi / 3, num_elements).reshape(shape)
    
            # Adding the tensor as an input
            angle_tensor1 = self.declare_variable('tensor', val=angle_val1)
    
            # Rotation in the x-axis for tensor1
            self.register_output('tensor_Rot_x',
                                 csdl.rotmat(angle_tensor1, axis='x'))
    
    
    rep = GraphRepresentation(ExampleSameRadianTensorRotX())
    sim = Simulator(rep)
    sim.run()
    
    print('tensor', sim['tensor'].shape)
    print(sim['tensor'])
    print('tensor_Rot_x', sim['tensor_Rot_x'].shape)
    print(sim['tensor_Rot_x'])
    
    return sim, rep