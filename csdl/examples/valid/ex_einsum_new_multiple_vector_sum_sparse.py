def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleMultipleVectorSumSparse(Model):
        def define(self):
    
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            self.register_output(
                'einsum_special2_sparse_derivs',
                csdl.einsum_new_api(vec,
                                    vec,
                                    operation=[
                                        (1, ),
                                        (2, ),
                                    ],
                                    partial_format='sparse'))
    
    
    rep = GraphRepresentation(ExampleMultipleVectorSumSparse())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_special2_sparse_derivs', sim['einsum_special2_sparse_derivs'].shape)
    print(sim['einsum_special2_sparse_derivs'])
    
    return sim, rep