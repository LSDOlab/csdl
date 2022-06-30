def example(Simulator):
    from csdl import Model, GraphRepresentation
    
    
    class ExampleComplex(Model):
        # Use a complex hierarchical model
        # return sim['x3_out'] = 1.0
        # return sim['x4_out'] = 10.0
        # return sim['hierarchical.ModelB.x2'] = 6.141
        # return sim['hierarchical.ModelB.ModelC.x1'] = 5.0
        # return sim['hierarchical.ModelB.ModelC.x2'] = 9.0
        # return sim['hierarchical.ModelF.x3_out'] = 0.01
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.hierarchical import Hierarchical
    
            self.create_input('y')
            self.add(
                Hierarchical(),
                name='hierarchical',
            )
    
    
    rep = GraphRepresentation(ExampleComplex())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep