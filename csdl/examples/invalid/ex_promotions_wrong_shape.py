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
    
    
    class ErrorWrongShape(Model):
        # Promotions should not be made if two variables with different shapes
        # return sim['f'] = 4
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            a = self.create_input('a', shape=(2, 1))
    
            # model should not auto-promote as 'f' and 'a' have different shapes
            self.add(AdditionFunction(),
                     name='model')
    
            f = self.declare_variable('f', shape=(2, 1))
            self.register_output('y', f+a)
    
    
    sim = Simulator(ErrorWrongShape())
    sim.run()
    
    print('f', sim['f'].shape)
    print(sim['f'])
    