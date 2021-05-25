def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorVecDifferentShapes(Model):
        def define(self):
            m = 3
            n = 4
    
            # Shape of the vectors
            vec_shape = (m, )
    
            # Values for the two vectors
            vec1 = np.arange(m)
            vec2 = np.arange(n, 2 * n)
    
            # Adding the vectors and tensors to csdl
            vec1 = self.declare_variable('vec1', val=vec1)
            vec2 = self.declare_variable('vec2', val=vec2)
    
            # Vector-Vector Dot Product
            self.register_output('VecVecDot', csdl.dot(vec1, vec2))
    
    
    sim = Simulator(ErrorVecDifferentShapes())
    sim.run()
    