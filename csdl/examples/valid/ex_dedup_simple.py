def example(Simulator):
    from csdl import Model
    import csdl
    import numpy as np
    
    
    class ExampleSimple(Model):
        def define(self):
            x = self.declare_variable('x')
            y = self.declare_variable('y')
    
            a = x + y
            b = x + y
            c = 2 * a
            d = 2 * b
    
            self.register_output('c', c)
            self.register_output('d', d)
    
    
    sim = Simulator(ExampleSimple())
    sim.run()
    
    print('vec1', sim['vec1'].shape)
    print(sim['vec1'])
    print('vec2', sim['vec2'].shape)
    print(sim['vec2'])
    print('VecVecCross', sim['VecVecCross'].shape)
    print(sim['VecVecCross'])
    
    return sim