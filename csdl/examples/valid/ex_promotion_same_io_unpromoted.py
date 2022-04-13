def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    import numpy as np
    
    
    class ExampleSameIOUnpromoted(Model):
        # Can create two outputs in two models with same names if unpromoted
        #  sim['b'] should be = 2
    
    
        def define(self):
    
            a1 = self.create_input('a1')
    
            m = Model()
            a2 = m.create_input('a2')
            m.register_output('f', a2 + 1)
    
            self.add(m, promotes=['a2'])
            self.register_output('f', a1 + 1)
    
    
    sim = Simulator(ExampleSameIOUnpromoted())
    sim.run()
    
    print('b', sim['b'].shape)
    print(sim['b'])
    
    return sim