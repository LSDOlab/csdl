def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleConnection(Model):
        # Connecting promoted variables
        # sim['f'] = 5
        def define(self):
    
            a = self.create_input('a')
    
            m = Model()
            a1 = m.create_input('a1')
            m.register_output('b1', a1 + 3)
    
            self.add(m, name='A')
    
            b2 = self.declare_variable('b2')
            self.register_output('f', a + b2)
            self.connect('b1', 'b2')
    
    
    sim = Simulator(ExampleConnection())
    sim.run()
    
    print('f', sim['f'].shape)
    print(sim['f'])
    
    return sim