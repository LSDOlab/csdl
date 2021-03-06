def example(Simulator):
    from csdl import Model, NonlinearBlockGS, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ErrorExponentialConstantArrayVariableScalar(Model):
        def define(self):
            x = self.declare_variable('x', val=2.0)
            a = np.array([1,2,3])
            y = csdl.exp_a(a, x)
            self.register_output('y', y) 
    
    
    rep = GraphRepresentation(ErrorExponentialConstantArrayVariableScalar())
    sim = Simulator(rep)
    sim.run()
    