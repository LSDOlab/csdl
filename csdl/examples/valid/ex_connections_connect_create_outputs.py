def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleConnectCreateOutputs(Model):
        # Connections should work for concatenations
        # return sim['y'] = [11, 6]
    
        def define(self):
    
            a = self.create_input('a', val=5)
    
            b = self.declare_variable('b')
            self.connect('a', 'b')

            c = self.create_output('c', shape=(2, ))
            c[0] = b + a
            c[1] = a
    
            d = self.declare_variable('d', shape=(2, ))
            self.connect('c', 'd')
            self.register_output('y', d + np.ones((2, )))
    
    
    
    rep = GraphRepresentation(ExampleConnectCreateOutputs())
    rep.visualize_graph()
    sim = Simulator(rep)
    sim.run()
    
    
    return sim, rep