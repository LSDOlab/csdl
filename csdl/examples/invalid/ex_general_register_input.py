def example(Simulator):
    from csdl import Model
    
    
    class ErrorRegisterInput(Model):
        def define(self):
            a = self.declare_variable('a', val=10)
            # This will raise a TypeError
            self.register_output('a', a)
    
    
    sim = Simulator(ErrorRegisterInput())
    sim.run()
    