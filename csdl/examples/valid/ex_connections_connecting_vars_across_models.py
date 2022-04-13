def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleConnectingVarsAcrossModels(Model):
        # Connecting variables accross multiple models
        # return sim['f'] = 16
    
        def define(self):
    
            a = self.create_input('a', val=3)
    
            m1 = Model()
            a1 = m1.declare_variable('a1')  # connect to a
            m1.register_output('b1', a1+3.0)
            self.add(m1)
    
            m2 = Model()
            a2 = m2.declare_variable('a2')  # connect to b1
            m2.register_output('b2', a2+3.0)
            self.add(m2)
    
            m3 = Model()
            a3 = m3.declare_variable('a3')  # connect to b1
            a4 = m3.declare_variable('a4')  # connect to b2
            m3.register_output('b3', a3+a4)
            self.add(m3)
    
            b4 = self.declare_variable('b4')  # connect to b3
            self.register_output('f', b4+1)
    
            self.connect('a', 'a1')
            self.connect('b1', 'a2')
            self.connect('b1', 'a3')
            self.connect('b2', 'a4')
            self.connect('b3', 'b4')
    
    
    sim = Simulator(ExampleConnectingVarsAcrossModels())
    sim.run()
    
    return sim