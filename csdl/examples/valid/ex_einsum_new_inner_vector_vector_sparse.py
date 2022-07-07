def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleInnerVectorVectorSparse(Model):
        def define(self):
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            self.register_output(
                'einsum_inner1_sparse_derivs',
                csdl.einsum_new_api(vec,
                                    vec,
                                    operation=[(0, ), (0, )],
                                    partial_format='sparse'))
    
    
    rep = GraphRepresentation(ExampleInnerVectorVectorSparse())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_inner1_sparse_derivs', sim['einsum_inner1_sparse_derivs'].shape)
    print(sim['einsum_inner1_sparse_derivs'])
    
    return sim, rep