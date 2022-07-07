def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleNestedUnpromotedConnections(Model):
        # Connecting unpromoted variables
        # sim['y'] = 8
    
        def define(self):
    
            a = self.create_input('a')
    
            m1 = Model()
            a1 = m1.create_input('a1')
    
            m2 = Model()
            a2 = m2.create_input('a2')
    
            m3 = Model()
            a3 = m3.create_input('a3')
    
            m4 = Model()
            a4 = m4.create_input('a4')
    
            m4.register_output('b4', a4 + 3)
    
            m3.add(m4, name='m4', promotes=[])
            b4 = m3.declare_variable('b4_connect')
            m3.register_output('b3', a3 + b4)
    
            m2.add(m3, name='m3', promotes=[])
            b3 = m2.declare_variable('b3_connect')
            m2.register_output('b2', a2 + b3)
    
            m1.add(m2, name='m2', promotes=[])
            b2 = m1.declare_variable('b2_connect')
            m1.register_output('b1', a1 + b2)
    
            self.add(m1, name='m1', promotes=[])
            b1 = self.declare_variable('b1_connect')
    
            self.connect('m1.b1', 'b1_connect')
            self.connect('m1.m2.b2', 'm1.b2_connect')
            self.connect('m1.m2.m3.b3', 'm1.m2.b3_connect')
            self.connect('m1.m2.m3.m4.b4', 'm1.m2.m3.b4_connect')
            self.register_output('y', a + b1)
    
    
    rep = GraphRepresentation(ExampleNestedUnpromotedConnections())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep