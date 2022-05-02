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
    
    
    class ErrorPartialPromotion(Model):
        # Trying to promote multiple variables with some of them invalid will result in an error
        # Return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            a = self.create_input('a', val=3.0)
            self.register_output('f', a + 1.0)
    
            # We can promote a and b but trying to
            # promote f will result in an error as it is already an output
            # in the parent model
            self.add(AdditionFunction(),
                     promotes=['a', 'b', 'f'],
                     name='model')
    
    
    rep = GraphRepresentation(ErrorPartialPromotion())
    sim = Simulator(rep)
    sim.run()
    