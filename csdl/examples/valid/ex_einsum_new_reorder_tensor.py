def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleReorderTensor(Model):
        def define(self):
    
            # Shape of Tensor
            shape3 = (2, 4, 3)
            c = np.arange(24).reshape(shape3)
    
            # Declaring tensor
            tens = self.declare_variable('c', val=c)
    
            # reorder of a tensor
            self.register_output(
                'einsum_reorder2',
                csdl.einsum_new_api(
                    tens,
                    operation=[(33, 66, 99), (99, 66, 33)],
                ))
    
    
    rep = GraphRepresentation(ExampleReorderTensor())
    sim = Simulator(rep)
    sim.run()
    
    print('c', sim['c'].shape)
    print(sim['c'])
    print('einsum_reorder2', sim['einsum_reorder2'].shape)
    print(sim['einsum_reorder2'])
    
    return sim, rep