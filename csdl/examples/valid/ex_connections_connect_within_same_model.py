def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleConnectWithinSameModel(Model):
        # Connecting promoted variables
        # sim['y'] = 5
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            a = self.create_input('a')
            b = self.create_input('b', val=2.0)
            c = a + b
    
            f1 = self.declare_variable('f1')
            self.register_output('y', c + f1)
            self.connect('b', 'f1')
    
    
    rep = GraphRepresentation(ExampleConnectWithinSameModel())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep