def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleElementwise(Model):
    
        def define(self):
    
            m = 2
            n = 3
            # Shape of the three tensors is (2,3)
            shape = (m, n)
    
            # Creating the values for two tensors
            val1 = np.array([[1, 5, -8], [10, -3, -5]])
            val2 = np.array([[2, 6, 9], [-1, 2, 4]])
    
            # Declaring the two input tensors
            tensor1 = self.declare_variable('tensor1', val=val1)
            tensor2 = self.declare_variable('tensor2', val=val2)
    
            # Creating the output for matrix multiplication
            ma = self.register_output('ElementwiseMax',
                                      csdl.max(tensor1, tensor2))
            assert ma.shape == (2, 3)
    
    
    rep = GraphRepresentation(ExampleElementwise())
    sim = Simulator(rep)
    sim.run()
    
    print('tensor1', sim['tensor1'].shape)
    print(sim['tensor1'])
    print('tensor2', sim['tensor2'].shape)
    print(sim['tensor2'])
    print('ElementwiseMax', sim['ElementwiseMax'].shape)
    print(sim['ElementwiseMax'])
    
    return sim, rep