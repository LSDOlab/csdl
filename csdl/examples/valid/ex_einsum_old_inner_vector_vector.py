def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleInnerVectorVector(Model):
        def define(self):
            a = np.arange(4)
    
            vec = self.declare_variable('a', val=a)
    
            # Inner Product of 2 vectors
            self.register_output('einsum_inner1',
                                 csdl.einsum(vec, vec, subscripts='i,i->'))
    
    
    rep = GraphRepresentation(ExampleInnerVectorVector())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_inner1', sim['einsum_inner1'].shape)
    print(sim['einsum_inner1'])
    
    return sim, rep