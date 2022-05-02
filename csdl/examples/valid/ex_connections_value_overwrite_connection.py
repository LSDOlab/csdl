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
    
    
    class ExampleValueOverwriteConnection(Model):
        # Connection should overwrite values
        # return sim['y'] = 6
    
        def define(self):
    
            a = self.create_input('a', val=3)
            b = self.declare_variable(
                'b',
                val=10)  # connect to a, value of 10 should be overwritten
    
            self.register_output('y', a + b)
    
            self.connect('a', 'b')
    
    
    rep = GraphRepresentation(ExampleValueOverwriteConnection())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep