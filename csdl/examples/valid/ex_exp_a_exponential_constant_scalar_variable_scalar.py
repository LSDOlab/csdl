def example(Simulator):
    from csdl import Model, NonlinearBlockGS
    import csdl
    import numpy as np
    
    
    class ExampleExponentialConstantScalarVariableScalar(Model):
        def define(self):
            a = 5.0 
            x = self.declare_variable('x', val=3)
            y = csdl.exp_a(a, x)
            self.register_output('y', y)
    
    
    sim = Simulator(ExampleExponentialConstantScalarVariableScalar())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim