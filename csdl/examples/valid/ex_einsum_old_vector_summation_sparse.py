def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleVectorSummationSparse(Model):
        def define(self):
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            self.register_output(
                'einsum_summ1_sparse_derivs',
                csdl.einsum(vec, subscripts='i->', partial_format='sparse'))
    
    
    rep = GraphRepresentation(ExampleVectorSummationSparse())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_summ1_sparse_derivs', sim['einsum_summ1_sparse_derivs'].shape)
    print(sim['einsum_summ1_sparse_derivs'])
    
    return sim, rep