def example(Simulator):
    import numpy as np
    import csdl
    from csdl import Model, GraphRepresentation
    
    
    class ErrorIntegerOutOfRange(Model):
        def define(self):
            a = self.declare_variable('a', val=0)
            x = self.create_output('x', shape=(1, ))
            # This triggers an error
            x[1] = a
    
    
    rep = GraphRepresentation(ErrorIntegerOutOfRange())
    sim = Simulator(rep)
    sim.run()
    