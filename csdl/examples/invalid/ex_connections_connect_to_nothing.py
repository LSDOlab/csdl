def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ErrorConnectToNothing(Model):
        # Connections can't be made between variables not existing
        # return error
    
        def define(self):
    
            model = Model()
            a = model.create_input(
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
    
            a = self.declare_variable('model.a')
            self.register_output('f', a + 2.0)
            self.connect('model.a', 'c')
    
    
    rep = GraphRepresentation(ErrorConnectToNothing())
    sim = Simulator(rep)
    sim.run()
    