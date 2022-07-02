def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleValueOverwriteConnection(Model):
        # Connection should overwrite values
        # return sim['f'] = 14
    
        def define(self):
    
            model = Model()
            a = model.declare_variable(
                'a', val=4.0)  # connect to b, creating a cycle
            b = model.declare_variable(
                'b', val=3.0)  # connect to a, creating a cycle
            c = a * b
            model.register_output('y', a + c)
    
            self.add(
                model,
                promotes=[],
                name='model',
            )
    
            d = self.declare_variable('d', val=10.0)
            self.register_output('f', 2 * d)
            self.connect('model.y', 'd')
    
    
    rep = GraphRepresentation(ExampleValueOverwriteConnection())
    sim = Simulator(rep)
    sim.run()
    
    print('model.y', sim['model.y'].shape)
    print(sim['model.y'])
    
    return sim, rep