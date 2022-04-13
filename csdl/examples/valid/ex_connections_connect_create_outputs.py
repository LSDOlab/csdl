def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleConnectCreateOutputs(Model):
        # Connections should work for concatenations
        # return sim['f'] = [11, 6]
    
        def define(self):
    
            a = self.create_input('a', val=5)
    
            b = self.declare_variable('b')
            c = self.create_output('c', shape=(2,))
            c[0] = b + a
            c[1] = a
    
            d = self.declare_variable('d', shape=(2,))
            self.register_output('f', d + np.ones((2,)))
    
            self.connect('a', 'b')  # Connecting a to b
            self.connect('c', 'd')  # Connecting a to b
    
    
    sim = Simulator(ExampleConnectCreateOutputs())
    sim.run()
    
    return sim