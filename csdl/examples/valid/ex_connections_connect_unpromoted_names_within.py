def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleConnectUnpromotedNamesWithin(Model):
        # Connecting Unromoted variables
        # sim['y'] = 3
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            with self.create_submodel('A') as mm:
                c = mm.create_input('c')
                with mm.create_submodel('B', promotes=[]) as m:
                    a = m.create_input('a')
                    b = m.create_input('b')
                    m.register_output('f', a + b)
                d = mm.declare_variable('d')
                mm.connect('B.f', 'd')
                mm.register_output('f1', c + d)
    
            f1 = self.declare_variable('f1')
            e = self.create_input('e')
            self.register_output('y', e + f1)
    
       
    
    rep = GraphRepresentation(ExampleConnectUnpromotedNamesWithin())
    sim = Simulator(rep)
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim, rep
