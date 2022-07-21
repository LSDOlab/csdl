def example(Simulator):
    from csdl import Model, NonlinearBlockGS
    import csdl
    import numpy as np
    
    
    class ExampleExponentialConstantArrayVariableArray(Model):
        def define(self):
            x = self.declare_variable('x', val=np.array([1,2,3]))
            a = np.array([1,2,3])
            y = csdl.exp_a(a, x)
            self.register_output('y', y)
    
    
    sim = Simulator(ExampleExponentialConstantArrayVariableArray())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim