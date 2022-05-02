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
    
    
    class ErrorConnectingCyclicalVars(Model):
        # Cannot make connections that make cycles
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=3)
            b = self.declare_variable('b')
    
            c = a * b
    
            self.register_output('y',
                                 a + c)  # connect to b, creating a cycle
    
            self.connect('y', 'b')
    
    
    rep = GraphRepresentation(ErrorConnectingCyclicalVars())
    sim = Simulator(rep)
    sim.run()
    