def example(Simulator):
    import numpy as np
    import csdl
    from csdl import Model
    
    
    class ErrorIntegerReuse(Model):
        def define(self):
            a = self.declare_variable('a', val=4)
            b = self.declare_variable('b', val=3)
            x = self.create_output('x', shape=(2, ))
            x[0] = a
            x[1] = b
            y = self.create_output('y', shape=(2, ))
            y[0] = x[0]
            y[1] = x[0]
    
    
    sim = Simulator(ErrorIntegerReuse())
    sim.run()
    