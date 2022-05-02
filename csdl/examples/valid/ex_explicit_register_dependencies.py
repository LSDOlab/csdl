def example(Simulator):
    from csdl import Model, GraphRepresentation, NonlinearBlockGS
    import csdl
    import numpy as np
    from csdl.examples.models.product import Product
    
    
    class ExampleRegisterDependencies(Model):
        def define(self):
            a = self.declare_variable('a')
            b = self.declare_variable('b')
            c = self.declare_variable('c')
            d = self.declare_variable('d')
            x = a + b  # 2
            y = c + d  # 2
            z = x * y  # 4
            self.register_output('z', z)
            self.register_output('x', x)
            self.register_output('y', y)
    
    
    rep = GraphRepresentation(ExampleRegisterDependencies())
    sim = Simulator(rep)
    sim.run()
    
    print('x', sim['x'].shape)
    print(sim['x'])
    print('y', sim['y'].shape)
    print(sim['y'])
    print('z', sim['z'].shape)
    print(sim['z'])
    
    return sim, rep