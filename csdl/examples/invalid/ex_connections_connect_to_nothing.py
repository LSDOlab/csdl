def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from connection_error import ConnectWithin
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.false_cycle import FalseCyclePost
    from csdl.examples.models.addition import AdditionFunction
    
    
    class ErrorConnectToNothing(Model):
        # Connections can't be made between variables not existing
        # return error
    
        def define(self):
    
            a = self.create_input('a')
    
            self.register_output('f', a + 2.0)
    
            self.connect(
                'a',
                'b')  # Connecting to a non-existing variable is an error
    
    
    rep = GraphRepresentation(ErrorConnectToNothing())
    sim = Simulator(rep)
    sim.run()
    