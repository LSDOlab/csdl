def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleScalarRotY(Model):
        def define(self):
            angle_val3 = np.pi / 3
    
            angle_scalar = self.declare_variable('scalar', val=angle_val3)
    
            # Rotation in the y-axis for scalar
            self.register_output('scalar_Rot_y', csdl.rotmat(angle_scalar,
                                                             axis='y'))
    
    
    rep = GraphRepresentation(ExampleScalarRotY())
    sim = Simulator(rep)
    sim.run()
    
    print('scalar', sim['scalar'].shape)
    print(sim['scalar'])
    print('scalar_Rot_y', sim['scalar_Rot_y'].shape)
    print(sim['scalar_Rot_y'])
    
    return sim, rep