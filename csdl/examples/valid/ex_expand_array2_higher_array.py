def example(Simulator):
    import csdl
    from csdl import Model, GraphRepresentation
    import numpy as np
    
    
    class ExampleArray2HigherArray(Model):
        def define(self):
            # Expanding an array into a higher-rank array
            val = np.array([
                [1., 2., 3.],
                [4., 5., 6.],
            ])
            array = self.declare_variable('array', val=val)
            expanded_array = csdl.expand(array, (2, 4, 3, 1), 'ij->iajb')
            self.register_output('expanded_array', expanded_array)
    
    
    rep = GraphRepresentation(ExampleArray2HigherArray())
    sim = Simulator(rep)
    sim.run()
    
    print('array', sim['array'].shape)
    print(sim['array'])
    print('expanded_array', sim['expanded_array'].shape)
    print(sim['expanded_array'])
    
    return sim, rep