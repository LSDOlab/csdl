def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleMatVecProduct(Model):
        def define(self):
            m = 3
            n = 2
            p = 4
    
            # Shape of the first matrix (3,2)
            shape1 = (m, n)
    
            # Shape of the second matrix (2,4)
            shape2 = (n, p)
    
            # Creating the values of both matrices
            val1 = np.arange(m * n).reshape(shape1)
    
            # Creating the values for the vector
            val3 = np.arange(n)
    
            # Declaring the two input matrices as mat1 and mat2
            mat1 = self.declare_variable('mat1', val=val1)
    
            # Declaring the input vector of size (n,)
            vec1 = self.declare_variable('vec1', val=val3)
    
            # Creating the output for a matrix multiplied by a vector
            self.register_output('MatVec', csdl.matmat(mat1, vec1))
    
    
    sim = Simulator(ExampleMatVecProduct())
    sim.run()
    
    print('mat1', sim['mat1'].shape)
    print(sim['mat1'])
    print('vec1', sim['vec1'].shape)
    print(sim['vec1'])
    print('MatVec', sim['MatVec'].shape)
    print(sim['MatVec'])
    
    return sim