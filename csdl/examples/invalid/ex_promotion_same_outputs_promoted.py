def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    import numpy as np
    
    
    class ErrorSameOutputsPromoted(Model):
        # Can't create two outputs in two models with same names
        # Return error
    
        def define(self):
    
            a1 = self.create_input('a1')
    
            m = Model()
            a2 = m.create_input('a2')
            m.register_output('f', a2 + 1)
    
            self.add(m)
            self.register_output('f', a1 + 1)
    
    
    sim = Simulator(ErrorSameOutputsPromoted())
    sim.run()
    