def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleSingleMatrixAlong1Random(Model):
    
        def define(self):
            n = 3
            m = 6
            np.random.seed(0)
    
            # Declare a matrix of shape 3x6 as input
            M1 = self.declare_variable('M1',
                                       val=np.random.rand(n * m).reshape(
                                           (n, m)))
    
            # Output the axiswise sum of matrix M1 along the columns
            self.register_output('single_matrix_sum_along_1',
                                 csdl.sum(M1, axes=(1, )))
    
    
    rep = GraphRepresentation(ExampleSingleMatrixAlong1Random())
    sim = Simulator(rep)
    sim.run()
    
    print('M1', sim['M1'].shape)
    print(sim['M1'])
    print('single_matrix_sum_along_1', sim['single_matrix_sum_along_1'].shape)
    print(sim['single_matrix_sum_along_1'])
    
    return sim, rep