def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ErrorCycleTwoModels(Model):
        # Promotions cannot be made if connections cause cycles
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
            from csdl.examples.models.subtraction import SubtractionFunction
    
            self.add(AdditionFunction(),
                     promotes=['b', 'a', 'f'],
                     name='model')
    
            self.add(SubtractionFunction(),
                     promotes=['c', 'd', 'f'],
                     name='model2')
    
            d = self.declare_variable('d')
            self.register_output(
                'b', d + 1.0)  # b is an input to 'model', creating a cycle
    
    
    rep = GraphRepresentation(ErrorCycleTwoModels())
    sim = Simulator(rep)
    sim.run()
    