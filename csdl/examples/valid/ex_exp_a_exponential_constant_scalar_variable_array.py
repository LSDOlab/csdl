def example(Simulator):
    from csdl import Model, NonlinearBlockGS, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleExponentialConstantScalarVariableArray(Model):
        def define(self):
            a = 5.0
            x = self.declare_variable('x', val=np.array([1,2,3]))
            y = csdl.exp_a(a, x)
            self.register_output('y', y)
    
    
    rep = GraphRepresentation(ExampleExponentialConstantScalarVariableArray())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep