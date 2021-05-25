def example(Simulator):
    import numpy as np
    from csdl import Model
    import csdl
    
    
    class ExampleInnerTensorVectorSparse(Model):
        def define(self):
    
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            # Shape of Tensor
            shape3 = (2, 4, 3)
            c = np.arange(24).reshape(shape3)
    
            # Declaring tensor
            tens = self.declare_variable('c', val=c)
    
            self.register_output(
                'einsum_inner2_sparse_derivs',
                csdl.einsum_new_api(tens,
                                    vec,
                                    operation=[
                                        ('rows', 0, 1),
                                        (0, ),
                                        ('rows', 1),
                                    ],
                                    partial_format='sparse'))
    
    
    sim = Simulator(ExampleInnerTensorVectorSparse())
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('c', sim['c'].shape)
    print(sim['c'])
    print('einsum_inner2_sparse_derivs', sim['einsum_inner2_sparse_derivs'].shape)
    print(sim['einsum_inner2_sparse_derivs'])
    
    return sim