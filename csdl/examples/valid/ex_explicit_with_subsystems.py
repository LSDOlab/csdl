def example(Simulator):
    from csdl import Model, NonlinearBlockGS
    import csdl
    import numpy as np
    
    
    class ExampleWithSubsystems(Model):
        def define(self):
            # Create input to main model
            x1 = self.create_input('x1', val=40)
    
            # Powers
            y4 = x1**2
    
            # Create subsystem that depends on previously created
            # input to main model
            m = Model()
    
            # This value is overwritten by connection from the main model
            a = m.declare_variable('x1', val=2)
            b = m.create_input('x2', val=12)
            m.register_output('prod', a * b)
            self.add(m, name='subsystem')
    
            # declare inputs with default values
            # This value is overwritten by connection
            # from the submodel
            x2 = self.declare_variable('x2', val=3)
    
            # Simple addition
            y1 = x2 + x1
            self.register_output('y1', y1)
    
            # Simple subtraction
            self.register_output('y2', x2 - x1)
    
            # Simple multitplication
            self.register_output('y3', x1 * x2)
    
            # Powers
            y5 = x2**2
    
            # register outputs in reverse order to how they are defined
            self.register_output('y5', y5)
            self.register_output('y6', y1 + y5)
            self.register_output('y4', y4)
    
    
    sim = Simulator(ExampleWithSubsystems())
    sim.run()
    
    print('prod', sim['prod'].shape)
    print(sim['prod'])
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
    
    return sim