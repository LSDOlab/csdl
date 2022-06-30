def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ExampleStackedModels(Model):
        # Autopromotions should work for models within models
        # return sim['b'] = 11
    
        def define(self):
    
            a = self.create_input('a', val=3.0)
    
            with self.create_submodel('model1') as m:
                am = m.create_input('am', val=2.0)
                with m.create_submodel('model2') as mm:
                    amm = mm.declare_variable(
                        'a', val=1.0
                    )  # 'model1.model2.a' should automatically promote and connect to 'a'
                    mm.register_output('bmm', amm * 2.0)
                bmm = m.declare_variable('bmm')
                m.register_output('bm', bmm + am)
    
            bm = self.declare_variable('bm')
            self.register_output('b', bm + a)
    
    
    rep = GraphRepresentation(ExampleStackedModels())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep