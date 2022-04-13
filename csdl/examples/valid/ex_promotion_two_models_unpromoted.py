def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    import numpy as np
    
    
    class ExampleTwoModelsUnpromoted(Model):
        # Can have two models with same variable names if  unpromoted
        # Return no error
    
        def define(self):
    
            m1 = Model()
            a1 = m1.create_input('a1')
            m1.register_output('f', a1 + 1)
    
            m2 = Model()
            a2 = m2.create_input('a1')
            m2.register_output('f', a2 + 1)
            self.add(m1)
            self.add(m2, promotes=[])
    
    
    sim = Simulator(ExampleTwoModelsUnpromoted())
    sim.run()
    
    return sim