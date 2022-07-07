def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ErrorDifferentShapes(Model):
        def define(self):
            # Creating two tensors
            shape1 = (2, 5, 4, 3)
            shape2 = (7, 5, 6, 3)
            num_elements1 = np.prod(shape1)
            num_elements2 = np.prod(shape2)
    
            tenval1 = np.arange(num_elements1).reshape(shape1)
            tenval2 = np.arange(num_elements2).reshape(shape2) + 6
    
            ten1 = self.declare_variable('ten1', val=tenval1)
            ten2 = self.declare_variable('ten2', val=tenval2)
    
            # Tensor-Tensor Dot Product specifying the last axis
            self.register_output('TenTenCross', csdl.cross(ten1, ten2, axis=3))
    
    
    rep = GraphRepresentation(ErrorDifferentShapes())
    sim = Simulator(rep)
    sim.run()
    