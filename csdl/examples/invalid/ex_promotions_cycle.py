def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ErrorCycle(Model):
        # Promotions cannot be made if connections cause cycles
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            g = self.create_input('g', val=3.0)
            f = self.declare_variable('f')
            b = self.register_output('b', f + g)
    
            # Add a model that outputs model.f and takes in g+f
            # Promoting b will create a cycle which throws an error
            self.add(AdditionFunction(),
                     promotes=['a', 'b', 'f'],
                     name='model')
    
            self.register_output('y', g + b)
    
    
    rep = GraphRepresentation(ErrorCycle())
    sim = Simulator(rep)
    sim.run()
    