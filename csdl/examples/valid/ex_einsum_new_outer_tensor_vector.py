def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleOuterTensorVector(Model):
        def define(self):
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            # Shape of Tensor
            shape3 = (2, 4, 3)
            c = np.arange(24).reshape(shape3)
    
            # Declaring tensor
            tens = self.declare_variable('c', val=c)
    
            # Outer Product of a tensor and a vector
            self.register_output(
                'einsum_outer2',
                csdl.einsum_new_api(
                    tens,
                    vec,
                    operation=[(0, 1, 30), (2, ), (0, 1, 30, 2)],
                ))
    
    
    rep = GraphRepresentation(ExampleOuterTensorVector())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('c', sim['c'].shape)
    print(sim['c'])
    print('einsum_outer2', sim['einsum_outer2'].shape)
    print(sim['einsum_outer2'])
    
    return sim, rep