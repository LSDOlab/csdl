def example(Simulator):
    from csdl import Model, NonlinearBlockGS
    import csdl
    import numpy as np
    
    
    class ExampleNoRegisteredOutput(Model):
        def define(self):
            model = Model()
            a = model.declare_variable('a', val=2)
            b = model.create_input('b', val=12)
            model.register_output('prod', a * b)
            self.add(model, name='sys', promotes=['*'])
    
            # These expressions do not lead to constructing any Component
            # objects
            x1 = self.declare_variable('x1')
            x2 = self.declare_variable('x2')
            y1 = x2 + x1
            y2 = x2 - x1
            y3 = x1 * x2
            y5 = x2**2
    
    
    sim = Simulator(ExampleNoRegisteredOutput())
    sim.run()
    
    print('prod', sim['prod'].shape)
    print(sim['prod'])
    
    return sim