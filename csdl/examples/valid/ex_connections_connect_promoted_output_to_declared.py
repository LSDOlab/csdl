def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleConnectPromotedOutputToDeclared(Model):
        # Connecting promoted variables
        # sim['y'] = 3
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            a = self.create_input('a')
    
            self.add(AdditionFunction(), name='A')
    
            f1 = self.declare_variable('f1')
            self.register_output('y', a + f1)
            self.connect('f', 'f1')
    
    
    rep = GraphRepresentation(ExampleConnectPromotedOutputToDeclared())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep