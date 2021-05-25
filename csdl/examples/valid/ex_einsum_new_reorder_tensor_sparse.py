def example(Simulator):
    import numpy as np
    from csdl import Model
    import csdl
    
    
    class ExampleReorderTensorSparse(Model):
        def define(self):
    
            # Shape of Tensor
            shape3 = (2, 4, 3)
            c = np.arange(24).reshape(shape3)
    
            # Declaring tensor
            tens = self.declare_variable('c', val=c)
    
            self.register_output(
                'einsum_reorder2_sparse_derivs',
                csdl.einsum_new_api(tens,
                                    operation=[(33, 66, 99), (99, 66, 33)],
                                    partial_format='sparse'))
    
    
    sim = Simulator(ExampleReorderTensorSparse())
    sim.run()
    
    print('c', sim['c'].shape)
    print(sim['c'])
    print('einsum_reorder2_sparse_derivs', sim['einsum_reorder2_sparse_derivs'].shape)
    print(sim['einsum_reorder2_sparse_derivs'])
    
    return sim