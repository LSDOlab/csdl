def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorMultiInputsAndAxis(Model):
        def define(self):
            # Creating the values for two tensors
            val1 = np.array([[1, 5, -8], [10, -3, -5]])
            val2 = np.array([[2, 6, 9], [-1, 2, 4]])
    
            # Declaring the two input tensors
            tensor1 = self.declare_variable('tensor1', val=val1)
            tensor2 = self.declare_variable('tensor2', val=val2)
    
            # Creating the output for matrix multiplication
            self.register_output('ElementwiseMinWithAxis',
                                 csdl.max(tensor1, tensor2, axis=0))
    
    
    sim = Simulator(ErrorMultiInputsAndAxis())
    sim.run()
    