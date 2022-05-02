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
    
    
    class ErrorTwoWayConnection(Model):
        # Cannot make two connections between two variables
        # return error
    
        def define(self):
    
            a = self.create_input('a',
                                  val=3)  # connect to b, creating a cycle
            b = self.declare_variable('b')  # connect to a, creating a cycle
    
            c = a * b
    
            self.register_output('f', a + c)
    
            self.connect('a', 'b')
            self.connect('b', 'a')
    
    
    rep = GraphRepresentation(ErrorTwoWayConnection())
    sim = Simulator(rep)
    sim.run()
    