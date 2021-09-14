def example(Simulator):
    import numpy as np
    import csdl
    from csdl import Model
    
    
    class ExampleInteger(Model):
        def define(self):
            a = self.declare_variable('a', val=0)
            b = self.declare_variable('b', val=1)
            c = self.declare_variable('c', val=2)
            d = self.declare_variable('d', val=7.4)
            e = self.declare_variable('e', val=np.pi)
            f = self.declare_variable('f', val=9)
            g = e + f
            x = self.create_output('x', shape=(7, ))
            x[0] = a
            x[1] = b
            x[2] = c
            x[3] = d
            x[4] = e
            x[5] = f
            x[6] = g
    
            # Get value from indices
            self.register_output('x0', x[0])
            self.register_output('x6', x[6])
            self.register_output('x_2', x[-2])
    
    
    sim = Simulator(ExampleInteger())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('x0', sim['x0'].shape)
    print(sim['x0'])
    print('x6', sim['x6'].shape)
    print(sim['x6'])
    
    return sim