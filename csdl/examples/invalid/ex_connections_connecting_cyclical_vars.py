def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorConnectingCyclicalVars(Model):
        # Cannot make connections that make cycles
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=3)
            b = self.declare_variable('b')
    
            c = a*b
    
            self.register_output('f', a+c)  # connect to b, creating a cycle
    
            self.connect('f', 'b')
    
    
    sim = Simulator(ErrorConnectingCyclicalVars())
    sim.run()
    