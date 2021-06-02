def example(Simulator):
    import numpy as np
    from csdl import Model
    import csdl
    
    
    class ExampleVectorSummation(Model):
        def define(self):
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            # Summation of all the entries of a vector
            self.register_output('einsum_summ1',
                                 csdl.einsum(
                                     vec,
                                     subscripts='i->',
                                 ))
    
    
    sim = Simulator(ExampleVectorSummation())
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_summ1', sim['einsum_summ1'].shape)
    print(sim['einsum_summ1'])
    
    return sim