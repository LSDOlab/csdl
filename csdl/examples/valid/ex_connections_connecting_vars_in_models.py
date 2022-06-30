def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleConnectingVarsInModels(Model):
        # Cannot make connections within models
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.connect_within import ConnectWithin
    
            self.add(ConnectWithin(
            ))  # Adding a connection within a model will throw an error
    
    
    rep = GraphRepresentation(ExampleConnectingVarsInModels())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep