def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    import numpy as np
    
    
    class ExampleUnconnectedVars(Model):
        # Declared variable with same local name should not be connected if unpromoted
        # sim['b'] should be = 2
    
        def define(self):
    
            m = Model()
            a = m.create_input('a', val=3.0)
            m.register_output('f', a + 1)
    
            self.add(m, promotes=[])
    
            a1 = self.declare_variable('a')
    
            self.register_output('b', a1+1.0)
    
    
    sim = Simulator(ExampleUnconnectedVars())
    sim.run()
    
    print('b', sim['b'].shape)
    print(sim['b'])
    
    return sim