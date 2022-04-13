def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    import numpy as np
    
    
    class ErrorStackedModels(Model):
        # Promotions should work for models within models
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=3.0)
    
            m = Model()
            am = m.create_input('am', val=2.0)
            mm = Model()
            amm = mm.create_input('a', val=1.0)
            mm.register_output('bmm', amm*2.0)
            m.add(mm)
            bmm = m.declare_variable('bmm')
            m.register_output('bm', bmm+am)
            self.add(m)
    
            bm = self.declare_variable('bm')
            self.register_output('b', bm+a)
    
    
    sim = Simulator(ErrorStackedModels())
    sim.run()
    