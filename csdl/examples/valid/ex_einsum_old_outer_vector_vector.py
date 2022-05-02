def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleOuterVectorVector(Model):
        def define(self):
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            # Outer Product of 2 vectors
            self.register_output('einsum_outer1',
                                 csdl.einsum(vec, vec, subscripts='i,j->ij'))
    
    
    rep = GraphRepresentation(ExampleOuterVectorVector())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_outer1', sim['einsum_outer1'].shape)
    print(sim['einsum_outer1'])
    
    return sim, rep