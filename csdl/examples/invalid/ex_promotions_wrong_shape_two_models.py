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
    
    
    class ErrorWrongShapeTwoModels(Model):
        # Promotions should not be made if two variables with different shapes
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
            from csdl.examples.models.subtraction import SubtractionVectorFunction
    
            a = self.create_input('a', val=3.0)
    
            input_dict = {}
            input_dict['name'] = 'a'
            # add two models where the second has the wrong shape
            self.add(AdditionFunction(),
                     name='model',
                     promotes=['a', 'b', 'f'])
    
            self.add(
                SubtractionVectorFunction(),
                name='model2',
                promotes=[
                    'f', 'c', 'd'
                ])  # promoting f will throw an error for shape mismatch
    
            d = self.declare_variable('d')
            self.register_output('y', d + a)
    
    
    rep = GraphRepresentation(ErrorWrongShapeTwoModels())
    sim = Simulator(rep)
    sim.run()
    