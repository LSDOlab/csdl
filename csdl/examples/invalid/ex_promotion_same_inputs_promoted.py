def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    import numpy as np
    
    
    class ErrorSameInputsPromoted(Model):
        # Can't create two inputs in two models with same names
        # Return error
    
        def define(self):
    
            a1 = self.create_input('a')
    
            m = Model()
            a2 = m.create_input('a')
            m.register_output('f1', a2 + 1)
    
            self.register_output('f2', a1 + 1)
    
            self.add(m)
    
    
    sim = Simulator(ErrorSameInputsPromoted())
    sim.run()
    