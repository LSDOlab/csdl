def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleMultipleMatrixAlong0(Model):
        def define(self):
            n = 3
            m = 6
    
            # Declare a matrix of shape 3x6 as input
            M1 = self.declare_variable('M1', val=np.arange(n * m).reshape((n, m)))
    
            # Declare another matrix of shape 3x6 as input
            M2 = self.declare_variable('M2',
                                       val=np.arange(n * m, 2 * n * m).reshape(
                                           (n, m)))
    
            # Output the elementwise average of the axiswise average of matrices M1 ad M2 along the columns
            self.register_output('multiple_matrix_average_along_0',
                                 csdl.average(M1, M2, axes=(0, )))
    
    
    sim = Simulator(ExampleMultipleMatrixAlong0())
    sim.run()
    
    print('M1', sim['M1'].shape)
    print(sim['M1'])
    print('M2', sim['M2'].shape)
    print(sim['M2'])
    print('multiple_matrix_average_along_0', sim['multiple_matrix_average_along_0'].shape)
    print(sim['multiple_matrix_average_along_0'])
    
    return sim