def example(Simulator):
    from csdl import Model
    import numpy as np
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.connect_within import ConnectWithin
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction, ParallelAdditionFunction
    
    
    class ExampleConnectIndex(Model):
        # Connections can't be made between variables that are already connected
        # return sim['y'] = np.array([[4.], [6.]])
    
        def define(self):
    
            a = self.create_input('a', val=np.array([[1.0], [2.0], [3.0]]))
            b = self.create_input('b', val=np.array([[3.0], [4.0]]))
            a_indexed = a[0:2, 0]  # == [[1.], [2.]]
            self.register_output('a_indexed', a_indexed)
    
            c = self.declare_variable('c', shape=(2, 1))
            self.register_output('y', c + b)
    
            self.connect('a_indexed', 'c')  # Connect a slice of a to variable c. Remember shapes must match
    
    
    sim = Simulator(ExampleConnectIndex())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim