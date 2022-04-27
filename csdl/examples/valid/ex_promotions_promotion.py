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
    
    
    class ExamplePromotion(Model):
        # Promoting a variable to a model will automatically connect them
        # Return sim['y'] = 7
    
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
                promotes=['a', 'f'],
                name='model')
    
            f = self.declare_variable('f')  # 'f' should automaticall connect to the 'f' in model
            self.register_output('y', f+a)
    
    
    sim = Simulator(ExamplePromotion())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim