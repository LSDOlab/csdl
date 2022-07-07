def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleSingleMatrix(Model):
    
        def define(self):
            n = 3
            m = 6
    
            # Declare a matrix of shape 3x6 as input
            M1 = self.declare_variable('M1',
                                       val=np.arange(n * m).reshape((n, m)))
    
            # Output the sum of all the elements of the tensor T1
            self.register_output('single_matrix_sum', csdl.sum(M1))
    
    
    rep = GraphRepresentation(ExampleSingleMatrix())
    sim = Simulator(rep)
    sim.run()
    
    print('M1', sim['M1'].shape)
    print(sim['M1'])
    print('single_matrix_sum', sim['single_matrix_sum'].shape)
    print(sim['single_matrix_sum'])
    
    return sim, rep