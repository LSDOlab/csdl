def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ErrorIncorrectAxisIndex(Model):
        def define(self):
            # Creating two tensors
            shape = (2, 5, 4, 3)
            num_elements = np.prod(shape)
    
            tenval1 = np.arange(num_elements).reshape(shape)
            tenval2 = np.arange(num_elements).reshape(shape) + 6
    
            ten1 = self.declare_variable('ten1', val=tenval1)
            ten2 = self.declare_variable('ten2', val=tenval2)
    
            # Tensor-Tensor Dot Product specifying the last axis
            self.register_output('TenTenCross', csdl.cross(ten1, ten2, axis=2))
    
    
    rep = GraphRepresentation(ErrorIncorrectAxisIndex())
    sim = Simulator(rep)
    sim.run()
    