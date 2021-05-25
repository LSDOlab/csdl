def example(Simulator):
    import numpy as np
    from csdl import Model
    import csdl
    
    
    class ExampleMultiplicationSum(Model):
        def define(self):
    
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            # Special operation: summation of all the entries of first
            # vector and scalar multiply the second vector with the computed
            # sum
            self.register_output(
                'einsum_special1',
                csdl.einsum_new_api(
                    vec,
                    vec,
                    operation=[(1, ), (2, ), (2, )],
                ))
    
    
    sim = Simulator(ExampleMultiplicationSum())
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_special1', sim['einsum_special1'].shape)
    print(sim['einsum_special1'])
    
    return sim