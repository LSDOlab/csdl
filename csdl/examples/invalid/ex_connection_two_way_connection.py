def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ErrorTwoWayConnection(Model):
        # Cannot make two connections between two variables
        # return error
    
        def define(self):
    
            a = self.create_input('a', val=3)  # connect to b, creating a cycle
            b = self.declare_variable('b')  # connect to a, creating a cycle
    
            c = a*b
    
            self.register_output('f', a+c)
    
            self.connect('a', 'b')
            self.connect('b', 'a')
    
    
    sim = Simulator(ErrorTwoWayConnection())
    sim.run()
    