def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleMultipleMatrixAlong0Random(Model):
    
        def define(self):
            n = 3
            m = 6
            np.random.seed(0)
    
            # Declare a matrix of shape 3x6 as input
            M1 = self.declare_variable('M1',
                                       val=np.random.rand(n * m).reshape(
                                           (n, m)))
    
            # Declare another matrix of shape 3x6 as input
            M2 = self.declare_variable('M2',
                                       val=np.random.rand(n * m).reshape(
                                           (n, m)))
    
            # Output the elementwise sum of the axiswise sum of matrices M1 ad M2 along the columns
            self.register_output('multiple_matrix_sum_along_0',
                                 csdl.sum(M1, M2, axes=(0, )))
    
    
    sim = Simulator(ExampleMultipleMatrixAlong0Random())
    sim.run()
    
    print('M1', sim['M1'].shape)
    print(sim['M1'])
    print('M2', sim['M2'].shape)
    print(sim['M2'])
    print('multiple_matrix_sum_along_0', sim['multiple_matrix_sum_along_0'].shape)
    print(sim['multiple_matrix_sum_along_0'])
    
    return sim