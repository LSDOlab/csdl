def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorConnectingUnpromotedNames(Model):
        # Cannot make connections using unpromoted names
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=3)
    
            m = Model()
            b = m.declare_variable('b')  # connect to a
            m.register_output('f', b+2)
            self.add(m, name='m')
    
            self.connect('a', 'm.b')
    
    
    sim = Simulator(ErrorConnectingUnpromotedNames())
    sim.run()
    