def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleConnectionIgnore(Model):
        # Connections should make 'ignored' variables 'unignored'
        # ****** NOT SURE IF THIS SHOULD RETURN AN ERROR OR NOT ******
        # return sim['f'] = 15
        def define(self):
    
            a = self.create_input('a', val=3)
            b = self.create_input('b', val=10)
            c_connect = b+2
    
            c = self.declare_variable('c')
            self.register_output('f', a + c)
    
            self.connect(c_connect.name, 'c')  # Not sure if this should throw an error or not...
    
    
    sim = Simulator(ExampleConnectionIgnore())
    sim.run()
    
    print('f', sim['f'].shape)
    print(sim['f'])
    
    return sim