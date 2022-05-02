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
    
    
    class ErrorConnectingWrongArgOrder(Model):
        # self.connect argument order must be in correct order
        # return error
    
        def define(self):
    
            a = self.create_input('a')
            b1 = self.declare_variable('b1')
    
            self.connect('b1', 'a')
            self.register_output('y', a + b1)
    
    
    rep = GraphRepresentation(ErrorConnectingWrongArgOrder())
    sim = Simulator(rep)
    sim.run()
    