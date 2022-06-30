def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ErrorInputs(Model):
        # Can't promote a variable to a model containing inputs with the same name
        # Return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            f = self.create_input('f', val=3.0)
    
            # Promoting b will raise an error as it is already an input
            self.add(AdditionFunction(),
                     promotes=['a', 'b', 'f'],
                     name='model')
    
            self.register_output('y', f + 1)
    
    
    rep = GraphRepresentation(ErrorInputs())
    sim = Simulator(rep)
    sim.run()
    