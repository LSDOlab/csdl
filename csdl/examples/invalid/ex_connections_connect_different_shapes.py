def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ErrorConnectDifferentShapes(Model):
        # Connections can't be made between two variables with different shapes
        # return error
    
        def define(self):
    
            d = self.create_input('d', shape=(2, 2))
    
            model = Model()
            a = model.declare_variable(
                'a', val=4.0)  # connect to b, creating a cycle
            b = model.declare_variable(
                'b', val=3.0)  # connect to a, creating a cycle
            c = a * b
            model.register_output('y', a + c)
    
            self.add(
                model,
                promotes=[],
                name='model',
            )
    
            y = self.declare_variable('model.y')
            self.register_output('f', y + 2.0)
            self.connect('d', 'model.a')
    
    
    rep = GraphRepresentation(ErrorConnectDifferentShapes())
    sim = Simulator(rep)
    sim.run()
    