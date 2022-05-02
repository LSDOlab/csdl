def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    from scipy.sparse import csc_matrix
    
    
    class ExampleMatVecProduct(Model):
        def define(self):
            m = 3
            n = 4
    
            # Shape of the first matrix (3,2)
            shape1 = (m, n)
    
            # Shape of the second matrix (2,4)
            shape2 = (n, )
    
            # Creating the values of both matrices
            val1 = np.arange(m * n).reshape(shape1)
            val2 = np.arange(n).reshape(shape2)
    
            # Declaring the input matrix and input vector
            mat1 = self.declare_variable('mat1', val=val1)
            vec1 = self.declare_variable('vec1', val=val2)
    
            # Creating the output for matrix-vector multiplication
            self.register_output('MatVec', csdl.matvec(mat1, vec1))
    
            sp = csc_matrix(mat1.val)
            self.register_output('SparseMatVec', csdl.matvec(sp, vec1))
    
    
    rep = GraphRepresentation(ExampleMatVecProduct())
    sim = Simulator(rep)
    sim.run()
    
    print('mat1', sim['mat1'].shape)
    print(sim['mat1'])
    print('vec1', sim['vec1'].shape)
    print(sim['vec1'])
    print('MatVec', sim['MatVec'].shape)
    print(sim['MatVec'])
    print('SparseMatVec', sim['SparseMatVec'].shape)
    print(sim['SparseMatVec'])
    
    return sim, rep