def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleValueOverwriteConnection(Model):
        # Connection should overwrite values
        # return sim[f] = 6
        def define(self):
    
            a = self.create_input('a', val=3)
            b = self.declare_variable('b', val=10)  # connect to a, value of 10 should be overwritten
    
            self.register_output('f', a + b)
    
            self.connect('a', 'b')
    
    
    sim = Simulator(ExampleValueOverwriteConnection())
    sim.run()
    
    print('f', sim['f'].shape)
    print(sim['f'])
    
    return sim