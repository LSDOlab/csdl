def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleMultiplicationSumSparse(Model):
        def define(self):
    
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            self.register_output(
                'einsum_special1_sparse_derivs',
                csdl.einsum_new_api(vec,
                                    vec,
                                    operation=[(1, ), (2, ), (2, )],
                                    partial_format='sparse'))
    
    
    rep = GraphRepresentation(ExampleMultiplicationSumSparse())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_special1_sparse_derivs', sim['einsum_special1_sparse_derivs'].shape)
    print(sim['einsum_special1_sparse_derivs'])
    
    return sim, rep