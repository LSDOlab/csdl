def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleConnectingUnpromotedNames(Model):
        # Cannot make connections using unpromoted names
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            c = self.create_input('c', val=3)
    
            self.add(AdditionFunction(), name='m')
    
            self.connect('c', 'm.b')
    
    
    rep = GraphRepresentation(ExampleConnectingUnpromotedNames())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep