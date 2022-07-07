def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleMultipleVectorSum(Model):
        def define(self):
    
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            # Special operation: sum all the entries of the first and second
            # vector to a single scalar
            self.register_output(
                'einsum_special2',
                csdl.einsum_new_api(vec, vec, operation=[(1, ), (2, )]))
    
    
    rep = GraphRepresentation(ExampleMultipleVectorSum())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('einsum_special2', sim['einsum_special2'].shape)
    print(sim['einsum_special2'])
    
    return sim, rep