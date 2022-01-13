def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleMultipleMatrixRandom(Model):
    
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
    
            # Output the elementwise sum of matrices M1 and M2
            self.register_output('multiple_matrix_sum', csdl.sum(M1, M2))
    
    
    sim = Simulator(ExampleMultipleMatrixRandom())
    sim.run()
    
    print('M1', sim['M1'].shape)
    print(sim['M1'])
    print('M2', sim['M2'].shape)
    print(sim['M2'])
    print('multiple_matrix_sum', sim['multiple_matrix_sum'].shape)
    print(sim['multiple_matrix_sum'])
    
    return sim