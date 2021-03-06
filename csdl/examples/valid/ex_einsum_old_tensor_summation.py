def example(Simulator):
    import numpy as np
    from csdl import Model, GraphRepresentation
    import csdl
    
    
    class ExampleTensorSummation(Model):
        def define(self):
            # Shape of Tensor
            shape3 = (2, 4, 3)
            c = np.arange(24).reshape(shape3)
    
            # Declaring tensor
            tens = self.declare_variable('c', val=c)
    
            # Summation of all the entries of a tensor
            self.register_output('einsum_summ2',
                                 csdl.einsum(
                                     tens,
                                     subscripts='ijk->',
                                 ))
    
    
    rep = GraphRepresentation(ExampleTensorSummation())
    sim = Simulator(rep)
    sim.run()
    
    print('c', sim['c'].shape)
    print(sim['c'])
    print('einsum_summ2', sim['einsum_summ2'].shape)
    print(sim['einsum_summ2'])
    
    return sim, rep