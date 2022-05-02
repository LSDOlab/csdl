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
    
    
    class ErrorTwoModelsPromoted(Model):
        # Can't have two models with same variable names if both promoted
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            self.add(AdditionFunction(), promotes=['a', 'f'], name='model')
    
            self.add(
                AdditionFunction(),
                promotes=[
                    'b', 'f'
                ],  # can't promote model2.f as we promoted model.f
                name='model2')
    
    
    rep = GraphRepresentation(ErrorTwoModelsPromoted())
    sim = Simulator(rep)
    sim.run()
    