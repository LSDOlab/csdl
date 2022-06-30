def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ErrorPromoteNonexistant(Model):
        # Try to promote a variable that does no exist
        # return keyerror
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            # a = self.create_input('a', val=3.0)
            a = self.declare_variable('a', val=3.0)
    
            self.add(
                AdditionFunction(),
                promotes=['a', 'f', 'c'],
                name='addition',
            )
    
            f = self.declare_variable('f')
            self.register_output('b', f + a)
    
    
    rep = GraphRepresentation(ErrorPromoteNonexistant())
    sim = Simulator(rep)
    sim.run()
    