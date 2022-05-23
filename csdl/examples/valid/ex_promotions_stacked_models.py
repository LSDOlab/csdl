def example(Simulator):
    import csdl
    from csdl import Model, GraphRepresentation
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.subtraction import SubtractionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.subtraction import SubtractionVectorFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.subtraction import SubtractionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    
    
    class ExampleStackedModels(Model):
        # Autopromotions should work for models within models
        # return sim['b'] = 11
    
        def define(self):
    
            a = self.create_input('a', val=3.0)
    
            m = Model()
            am = m.create_input('am', val=2.0)
            mm = Model()
            amm = mm.declare_variable(
                'a', val=1.0
            )  # 'model1.model2.a' should automatically promote and connect to 'a'
            mm.register_output('bmm', amm * 2.0)
            m.add(mm, name='model2')
            bmm = m.declare_variable('bmm')
            m.register_output('bm', bmm + am)
            self.add(m, name='model1')
    
            bm = self.declare_variable('bm')
            self.register_output('b', bm + a)
    
    
    rep = GraphRepresentation(ExampleStackedModels())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep