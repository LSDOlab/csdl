def example(Simulator):
    import csdl
    from csdl import Model
    import numpy as np
    
    
    class ErrorArrayInvalidIndices1(Model):
        def define(self):
            # Test array expansion
            val = np.array([
                [1., 2., 3.],
                [4., 5., 6.],
            ])
            array = self.declare_variable('array', val=val)
            expanded_array = csdl.expand(array, (2, 4, 3, 1), 'ij->iaj')
            self.register_output('expanded_array', expanded_array)
    
    
    sim = Simulator(ErrorArrayInvalidIndices1())
    sim.run()
    