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
    
    
    class ErrorConnectingTwoVariables(Model):
        # Connecting two variables to same the variable returns error
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=3)
            b = self.declare_variable('b')
            self.register_output('f', a + b)
    
            c = self.declare_variable('c')
            d = self.declare_variable('d')
            self.register_output('f2', c + d)
    
            self.connect('f', 'c')
            self.connect('a', 'c')
    
    
    rep = GraphRepresentation(ErrorConnectingTwoVariables())
    sim = Simulator(rep)
    sim.run()
    