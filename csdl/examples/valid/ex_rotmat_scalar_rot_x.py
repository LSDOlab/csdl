def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleScalarRotX(Model):
        def define(self):
            angle_val3 = np.pi / 3
    
            angle_scalar = self.declare_variable('scalar', val=angle_val3)
    
            # Rotation in the x-axis for scalar
            self.register_output('scalar_Rot_x', csdl.rotmat(angle_scalar,
                                                             axis='x'))
    
    
    rep = GraphRepresentation(ExampleScalarRotX())
    sim = Simulator(rep)
    sim.run()
    
    print('scalar', sim['scalar'].shape)
    print(sim['scalar'])
    print('scalar_Rot_x', sim['scalar_Rot_x'].shape)
    print(sim['scalar_Rot_x'])
    
    return sim, rep