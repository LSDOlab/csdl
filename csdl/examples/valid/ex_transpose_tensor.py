def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleTensor(Model):
        def define(self):
    
            # Declare tens as an input tensor with shape = (4, 3, 2, 5)
            tens = self.declare_variable(
                'Tens',
                val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
            )
    
            # Compute the transpose of tens
            self.register_output('tensor_transpose', csdl.transpose(tens))
    
    
    rep = GraphRepresentation(ExampleTensor())
    sim = Simulator(rep)
    sim.run()
    
    print('Tens', sim['Tens'].shape)
    print(sim['Tens'])
    print('tensor_transpose', sim['tensor_transpose'].shape)
    print(sim['tensor_transpose'])
    
    return sim, rep