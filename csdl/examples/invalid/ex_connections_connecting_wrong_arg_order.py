def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorConnectingWrongArgOrder(Model):
        # self.connect argument order must be in correct order
        # return error
    
        def define(self):
    
            a = self.create_input('a')
            b1 = self.declare_variable('b1')
    
            self.connect('b1', 'a')
            self.register_output('f', a + b1)
    
    
    sim = Simulator(ErrorConnectingWrongArgOrder())
    sim.run()
    