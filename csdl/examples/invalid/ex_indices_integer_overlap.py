def example(Simulator):
    import numpy as np
    import csdl
    from csdl import Model
    
    
    class ErrorIntegerOverlap(Model):
        def define(self):
            a = self.declare_variable('a', val=0)
            b = self.declare_variable('b', val=1)
            x = self.create_output('x', shape=(2, ))
            x[0] = a
            # This triggers an error
            x[0] = b
    
    
    sim = Simulator(ErrorIntegerOverlap())
    sim.run()
    