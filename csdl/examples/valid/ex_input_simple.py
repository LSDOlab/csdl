def example(Simulator):
    from csdl import Model, GraphRepresentation
    import numpy as np
    
    
    class ExampleSimple(Model):
        def define(self):
            z = self.create_input('z', val=10)
    
    
    rep = GraphRepresentation(ExampleSimple())
    sim = Simulator(rep)
    sim.run()
    
    print('z', sim['z'].shape)
    print(sim['z'])
    
    return sim, rep