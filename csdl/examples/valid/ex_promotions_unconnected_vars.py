def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ExampleUnconnectedVars(Model):
        # Declared variable with same local name should not be connected if unpromoted
        # sim['f'] should be = 2
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            self.add(AdditionFunction(), promotes=[], name='model')
    
            f = self.declare_variable('f')
    
            self.register_output('y', f + 1.0)
    
    
    rep = GraphRepresentation(ExampleUnconnectedVars())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep