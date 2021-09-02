def example(Simulator):
    import numpy as np
    import csdl
    from csdl import Model
    
    
    class ExampleOneDimensional(Model):
        def define(self):
            n = 20
            u = self.declare_variable('u',
                                      shape=(n, ),
                                      val=np.arange(n).reshape((n, )))
            v = self.declare_variable('v',
                                      shape=(n - 4, ),
                                      val=np.arange(n - 4).reshape((n - 4, )))
            w = self.declare_variable('w',
                                      shape=(4, ),
                                      val=16 + np.arange(4).reshape((4, )))
            x = self.create_output('x', shape=(n, ))
            x[0:n] = 2 * (u + 1)
            y = self.create_output('y', shape=(n, ))
            y[0:n - 4] = 2 * (v + 1)
            y[n - 4:n] = w - 3
    
            # Get value from indices
            z = self.create_output('z', shape=(3, ))
            z[0:3] = csdl.expand(x[2], (3, ))
            self.register_output('x0_5', x[0:5])
            self.register_output('x3_', x[3:])
            self.register_output('x2_4', x[2:4])
            self.register_output('x_last', x[-1])
    
    
    sim = Simulator(ExampleOneDimensional())
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('y', sim['y'].shape)
    print(sim['y'])
    print('z', sim['z'].shape)
    print(sim['z'])
    print('x0_5', sim['x0_5'].shape)
    print(sim['x0_5'])
    print('x3_', sim['x3_'].shape)
    print(sim['x3_'])
    print('x2_4', sim['x2_4'].shape)
    print(sim['x2_4'])
    print('x_last', sim['x_last'].shape)
    print(sim['x_last'])
    
    return sim