def example(Simulator):
    from csdl import Model
    import numpy as np
    
    
    class ExampleSimple(Model):
        def define(self):
            z = self.create_input('z', val=10)
    
    
    sim = Simulator(ExampleSimple())
    sim.run()
    
    print('z', sim['z'].shape)
    print(sim['z'])
    
    return sim