def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleInnerTensorVector(Model):
        def define(self):
            a = np.arange(4)
            vec = self.declare_variable('a', val=a)
    
            # Shape of Tensor
            shape3 = (2, 4, 3)
            c = np.arange(24).reshape(shape3)
    
            # Declaring tensor
            tens = self.declare_variable('c', val=c)
    
            # Inner Product of a tensor and a vector
            self.register_output(
                'einsum_inner2',
                csdl.einsum_new_api(
                    tens,
                    vec,
                    operation=[('rows', 0, 1), (0, ), ('rows', 1)],
                ))
    
    
    rep = GraphRepresentation(ExampleInnerTensorVector())
    sim = Simulator(rep)
    sim.run()
    
    print('a', sim['a'].shape)
    print(sim['a'])
    print('c', sim['c'].shape)
    print(sim['c'])
    print('einsum_inner2', sim['einsum_inner2'].shape)
    print(sim['einsum_inner2'])
    
    return sim, rep