def example(Simulator):
    from csdl import Model, GraphRepresentation, NonlinearBlockGS
    import csdl
    import numpy as np
    from csdl.examples.models.product import Product
    
    
    class ExampleNoRegisteredOutput(Model):
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.product import Product
    
            self.add(Product(), name='sys')
    
            # These expressions are not passed to the compiler back end
            x1 = self.declare_variable('x1')
            x2 = self.declare_variable('x2')
            y1 = x2 + x1
            y2 = x2 - x1
            y3 = x1 * x2
            y5 = x2**2
    
    
    rep = GraphRepresentation(ExampleNoRegisteredOutput())
    sim = Simulator(rep)
    sim.run()
    
    print('prod', sim['prod'].shape)
    print(sim['prod'])
    
    return sim, rep