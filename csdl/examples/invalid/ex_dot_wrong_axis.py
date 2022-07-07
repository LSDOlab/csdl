def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ErrorWrongAxis(Model):
        def define(self):
            m = 3
            n = 4
            p = 5
    
            # Shape of the tensors
            ten_shape1 = (m, n, p)
            ten_shape2 = (m, n, p)
    
            # Number of elements in the tensors
            num_ten_elements1 = np.prod(ten_shape1)
            num_ten_elements2 = np.prod(ten_shape2)
    
            # Values for the two tensors
            ten1 = np.arange(num_ten_elements1).reshape(ten_shape1)
            ten2 = np.arange(num_ten_elements2,
                             2 * num_ten_elements2).reshape(ten_shape2)
    
            ten1 = self.declare_variable('ten1', val=ten1)
            ten2 = self.declare_variable('ten2', val=ten2)
    
            # Tensor-Tensor Dot Product specifying the first axis
            self.register_output('TenTenDotFirst', csdl.dot(ten1, ten2, axis=1))
    
    
    rep = GraphRepresentation(ErrorWrongAxis())
    sim = Simulator(rep)
    sim.run()
    