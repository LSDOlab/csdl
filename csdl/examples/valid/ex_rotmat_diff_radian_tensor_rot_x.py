def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleDiffRadianTensorRotX(Model):
        def define(self):
    
            # Shape of a random tensor rotation matrix
            shape = (2, 3, 4)
    
            num_elements = np.prod(shape)
    
            # Vector of angles in radians
            angle_val2 = np.repeat(
                np.pi / 3, num_elements) + 2 * np.pi * np.arange(num_elements)
    
            angle_val2 = angle_val2.reshape(shape)
    
            # Adding the vector as an input
            angle_tensor = self.declare_variable('tensor', val=angle_val2)
    
            # Rotation in the x-axis for tensor2
            self.register_output('tensor_Rot_x', csdl.rotmat(angle_tensor,
                                                             axis='x'))
    
    
    rep = GraphRepresentation(ExampleDiffRadianTensorRotX())
    sim = Simulator(rep)
    sim.run()
    
    print('tensor', sim['tensor'].shape)
    print(sim['tensor'])
    print('tensor_Rot_x', sim['tensor_Rot_x'].shape)
    print(sim['tensor_Rot_x'])
    
    return sim, rep