def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleMultiplicationSum(Model):
        def define(self):
    
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            # Special operation: summation of all the entries of first
            # vector and scalar multiply the second vector with the computed
            # sum
            self.register_output('einsum_special1',
                                 csdl.einsum(vec, vec, subscripts='i,j->j'))
    
    
    rep = GraphRepresentation(ExampleMultiplicationSum())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_special1', sim['einsum_special1'].shape)
    print(sim['einsum_special1'])
    
    return sim, rep