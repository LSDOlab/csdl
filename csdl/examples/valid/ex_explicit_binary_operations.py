def example(Simulator):
    from csdl import Model, NonlinearBlockGS
    import csdl
    import numpy as np
    
    
    class ExampleBinaryOperations(Model):
        def define(self):
            # declare inputs with default values
            x1 = self.declare_variable('x1', val=2)
            x2 = self.declare_variable('x2', val=3)
            x3 = self.declare_variable('x3', val=np.arange(7))
    
            # Expressions with multiple binary operations
            y1 = -2 * x1**2 + 4 * x2 + 3
            self.register_output('y1', y1)
    
            # Elementwise addition
            y2 = x2 + x1
    
            # Elementwise subtraction
            y3 = x2 - x1
    
            # Elementwise multitplication
            y4 = x1 * x2
    
            # Elementwise division
            y5 = x1 / x2
            y6 = x1 / 3
            y7 = 2 / x2
    
            # Elementwise Power
            y8 = x2**2
            y9 = x1**2
    
            self.register_output('y2', y2)
            self.register_output('y3', y3)
            self.register_output('y4', y4)
            self.register_output('y5', y5)
            self.register_output('y6', y6)
            self.register_output('y7', y7)
            self.register_output('y8', y8)
            self.register_output('y9', y9)
    
            # Adding other expressions
            self.register_output('y10', y1 + y7)
    
            # Array with scalar power
            y11 = x3**2
            self.register_output('y11', y11)
    
            # Array with array of powers
            y12 = x3**(2 * np.ones(7))
            self.register_output('y12', y12)
    
    
    sim = Simulator(ExampleBinaryOperations())
    sim.run()
    
    print('y1', sim['y1'].shape)
    print(sim['y1'])
    print('y2', sim['y2'].shape)
    print(sim['y2'])
    print('y3', sim['y3'].shape)
    print(sim['y3'])
    print('y4', sim['y4'].shape)
    print(sim['y4'])
    print('y5', sim['y5'].shape)
    print(sim['y5'])
    print('y6', sim['y6'].shape)
    print(sim['y6'])
    print('y7', sim['y7'].shape)
    print(sim['y7'])
    print('y8', sim['y8'].shape)
    print(sim['y8'])
    print('y9', sim['y9'].shape)
    print(sim['y9'])
    print('y10', sim['y10'].shape)
    print(sim['y10'])
    print('y11', sim['y11'].shape)
    print(sim['y11'])
    print('y12', sim['y12'].shape)
    print(sim['y12'])
    
    return sim