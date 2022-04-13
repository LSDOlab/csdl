def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorConnectingVarsInModels(Model):
        # Cannot make connections within models
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=3)
    
            m = Model()
            a1 = m.declare_variable('a')
            a2 = m.declare_variable('a2')  # connect to a
    
            m.register_output('f', a1+a2)
            m.connect('a', 'a2')
    
            self.add(m)
    
    
    sim = Simulator(ErrorConnectingVarsInModels())
    sim.run()
    