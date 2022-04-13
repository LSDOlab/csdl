def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorConnectCreateOutputs(Model):
        # Connections should not work for concatenations if wrong shape
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=5)
    
            b = self.declare_variable('b')
            c = self.create_output('c', shape=(2,))
            c[0] = b + a
            c[1] = a
    
            d = self.declare_variable('d')
            self.register_output('f', d + np.ones((2,)))
    
            self.connect('a', 'b')  # Connecting a to b
            self.connect('c', 'd')  # Connecting c to d but d is wrong shape
    
    
    sim = Simulator(ErrorConnectCreateOutputs())
    sim.run()
    