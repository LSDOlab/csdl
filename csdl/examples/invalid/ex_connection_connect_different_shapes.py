def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorConnectDifferentShapes(Model):
        # Connections can't be made between two variables with different shapes
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=np.array([3, 3]))
            b = self.declare_variable('b', shape=(1,))
    
            self.register_output('f', b + 2.0)
    
            self.connect('a', 'b')  # Connecting wrong shapes should be an error
    
    
    sim = Simulator(ErrorConnectDifferentShapes())
    sim.run()
    