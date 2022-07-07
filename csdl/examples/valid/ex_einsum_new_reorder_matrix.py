def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleReorderMatrix(Model):
        def define(self):
            shape2 = (5, 4)
            b = np.arange(20).reshape(shape2)
            mat = self.declare_variable('b', val=b)
    
            # reorder of a matrix
            self.register_output(
                'einsum_reorder1',
                csdl.einsum_new_api(mat, operation=[(46, 99), (99, 46)]))
    
    
    rep = GraphRepresentation(ExampleReorderMatrix())
    sim = Simulator(rep)
    sim.run()
    
    print('b', sim['b'].shape)
    print(sim['b'])
    print('einsum_reorder1', sim['einsum_reorder1'].shape)
    print(sim['einsum_reorder1'])
    
    return sim, rep