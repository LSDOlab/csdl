def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ExampleSameIOUnpromoted(Model):
        # Can create two outputs in two models with same names if unpromoted
        # Return sim['f'] = 1, sim['model.f'] = 2
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            a = self.create_input('a')
    
            self.add(
                AdditionFunction(),
                promotes=[
                    'a', 'b'
                ],  # Promoting only a and b allows two outputs: f and model.f
                name='model')
    
            self.register_output('f', a * 1.0)
    
    
    rep = GraphRepresentation(ExampleSameIOUnpromoted())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep