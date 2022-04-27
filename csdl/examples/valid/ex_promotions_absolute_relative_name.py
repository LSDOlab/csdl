def example(Simulator):
    from csdl import Model
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
    
    
    class ExampleAbsoluteRelativeName(Model):
        # We can access absolute and relative names for variables interchangibly
        # Return sim['model.f'] = 4 and  sim['f'] = 4
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            a = self.create_input('a', val=3.0)
    
            self.add(
                AdditionFunction(),
                promotes=['a', 'b', 'f'],  # We can access f as sim['f']  or sim['model.f']
                name='model')
    
    
    sim = Simulator(ExampleAbsoluteRelativeName())
    sim.run()
    
    print('f', sim['f'].shape)
    print(sim['f'])
    print('model.f', sim['model.f'].shape)
    print(sim['model.f'])
    
    return sim