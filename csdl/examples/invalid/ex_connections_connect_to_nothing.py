def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorConnectToNothing(Model):
        # Connections can't be made between variables not existing
        # return error
    
        def define(self):
    
            a = self.create_input('a')
    
            self.register_output('f', a + 2.0)
    
            self.connect('a', 'b')  # Connecting to a non-existing variable is an error
    
    
    sim = Simulator(ErrorConnectToNothing())
    sim.run()
    