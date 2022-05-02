def example(Simulator):
    from csdl import Model, GraphRepresentation, NonlinearBlockGS
    import csdl
    import numpy as np
    from csdl.examples.models.product import Product
    
    
    class ExampleLiterals(Model):
        def define(self):
            x = self.declare_variable('x', val=3)
            y = -2 * x**2 + 4 * x + 3
            self.register_output('y', y)
    
    
    rep = GraphRepresentation(ExampleLiterals())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep