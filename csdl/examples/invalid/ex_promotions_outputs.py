def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ErrorOutputs(Model):
        # Can't promote a variable to a model containing outputs with the same name
        # Return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            b = self.create_input('b', val=3.0)
            self.register_output('f', b + 1)
    
            # Promoting f will raise an error because it is already an output
            self.add(AdditionFunction(),
                     promotes=['a', 'b', 'f'],
                     name='model')
    
    
    rep = GraphRepresentation(ErrorOutputs())
    sim = Simulator(rep)
    sim.run()
    