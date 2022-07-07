def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ErrorConnectingCyclicalVars(Model):
        # Cannot make connections that make cycles
        # return error
    
        def define(self):
    
            a = self.create_input('a')
    
            model = Model()
            a = model.declare_variable('a')
            b = model.declare_variable('b', val=3.0)
            c = a * b
            model.register_output('y',
                                  a + c)  # connect to b, creating a cycle
            self.add(model)
    
            self.connect('y', 'b')
    
    
    rep = GraphRepresentation(ErrorConnectingCyclicalVars())
    sim = Simulator(rep)
    sim.run()
    