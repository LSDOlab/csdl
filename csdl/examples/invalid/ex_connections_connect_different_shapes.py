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
    
    
    class ErrorConnectDifferentShapes(Model):
        # Connections can't be made between two variables with different shapes
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=np.array([3, 3]))
            b = self.declare_variable('b', shape=(1, ))
    
            self.register_output('f', b + 2.0)
    
            self.connect('a',
                         'b')  # Connecting wrong shapes should be an error
    
    
    rep = GraphRepresentation(ErrorConnectDifferentShapes())
    sim = Simulator(rep)
    sim.run()
    