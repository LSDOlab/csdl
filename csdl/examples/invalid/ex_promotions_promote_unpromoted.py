def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ErrorPromoteUnpromoted(Model):
        # Try to promote an unpromoted variable
        # return keyerror
    
        def define(self):
    
            a = self.create_input('a', val=3.0)
    
            m = Model()
            am = m.create_input('am', val=2.0)
            m.register_output('bm', am * 2.0)
    
            mm = Model()
            mm.create_input('b', val=2.0)
            # Do no promote b
            m.add(mm, name='model2', promotes=[])
    
            # but promote model2.b
            self.add(m, name='model1', promotes=['model2.b'])
    
            bb = self.declare_variable('b')
            self.register_output('f', bb + a)
    
    
    rep = GraphRepresentation(ErrorPromoteUnpromoted())
    sim = Simulator(rep)
    sim.run()
    