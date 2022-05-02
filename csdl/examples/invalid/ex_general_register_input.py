def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ErrorRegisterInput(Model):
        def define(self):
            a = self.declare_variable('a', val=10)
            # This will raise a TypeError
            self.register_output('a', a)
    
    
    rep = GraphRepresentation(ErrorRegisterInput())
    sim = Simulator(rep)
    sim.run()
    