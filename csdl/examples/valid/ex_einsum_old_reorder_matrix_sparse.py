def example(Simulator):
    import numpy as np
    from csdl import Model
    import csdl
    
    
    class ExampleReorderMatrixSparse(Model):
        def define(self):
    
            shape2 = (5, 4)
            b = np.arange(20).reshape(shape2)
            mat = self.declare_variable('b', val=b)
    
            self.register_output(
                'einsum_reorder1_sparse_derivs',
                csdl.einsum(
                    mat,
                    subscripts='ij->ji',
                    partial_format='sparse',
                ))
    
    
    sim = Simulator(ExampleReorderMatrixSparse())
    sim.run()
    
    print('b', sim['b'].shape)
    print(sim['b'])
    print('einsum_reorder1_sparse_derivs', sim['einsum_reorder1_sparse_derivs'].shape)
    print(sim['einsum_reorder1_sparse_derivs'])
    
    return sim