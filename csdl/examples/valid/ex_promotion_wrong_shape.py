def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    import numpy as np
    
    
    class ExampleWrongShape(csdl.Model):
        # Promotions should not automatically be made if two variables with different shapes
        # return sim['f'] = 4
    
        def define(self):
            import numpy as np
            a = self.create_input('a', val=3.0)
    
            m = csdl.Model()
            am = m.create_input('am', val=np.array([2.0, 3.0]))
            m.register_output('bm', am+np.array([2.0, 3.0]))
            self.add(m)  # should not auto promote as it would create namespace errors (?) with 'bm'
    
            bm = self.declare_variable('bm')
            self.register_output('f', bm+a)
    
    
    sim = Simulator(ExampleWrongShape())
    sim.run()
    
    print('f', sim['f'].shape)
    print(sim['f'])
    
    return sim