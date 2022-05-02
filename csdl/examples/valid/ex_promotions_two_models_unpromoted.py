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
    
    
    class ExampleTwoModelsUnpromoted(Model):
        # Can have two models with same variable names if only one variable is promoted
        # return sim['f'] = 3, sim['model2.f'] = 2
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            self.create_input('a', val=2.0)
            self.add(AdditionFunction(),
                     promotes=['a', 'b'
                               'f'],
                     name='model')
    
            self.add(
                AdditionFunction(),
                promotes=[],  # can't promote model2.f as we promoted model.f
                name='model2')
    
    
    rep = GraphRepresentation(ExampleTwoModelsUnpromoted())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep