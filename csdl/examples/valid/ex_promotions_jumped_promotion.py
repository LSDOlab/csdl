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
    from csdl.examples.models.subtraction import SubtractionFunction
    from csdl.examples.models.hierarchical import Hierarchical
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.subtraction import SubtractionVectorFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    
    
    class ExampleJumpedPromotion(Model):
        # Promote a variable that has not been explicitly declared in that model but in the model above
        # return sim['f'] = 5
    
        def define(self):
    
            a = self.create_input('a', val=3.0)
    
            m = Model()
            am = m.create_input('am', val=2.0)
            m.register_output('bm', am*2.0)
    
            mm = Model()
            mm.create_input(
                'b', val=2.0
            )
            # promote b in m even though it hasn't been declared in m
            m.add(mm, name='model2', promotes=['b'])
            self.add(m, name='model1', promotes=['b'])
    
            bb = self.declare_variable('b')
            self.register_output('f', bb + a)
    
    
    rep = GraphRepresentation(ExampleJumpedPromotion())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep