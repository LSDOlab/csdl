def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ErrorTwoWayConnection(Model):
        # Cannot make two connections between two variables
        # return error
    
        def define(self):
    
            model = Model()
            a = model.declare_variable(
                'a', val=2.0)  # connect to b, creating a cycle
            b = model.declare_variable(
                'b', val=2.0)  # connect to a, creating a cycle
            c = a * b
            model.register_output('y', a + c)
    
            self.add(model, promotes=[], name='model')
            self.connect('model.a', 'model.b')
            self.connect('model.b', 'model.a')
    
    
    rep = GraphRepresentation(ErrorTwoWayConnection())
    sim = Simulator(rep)
    sim.run()
    