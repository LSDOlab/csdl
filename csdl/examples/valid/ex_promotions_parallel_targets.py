def example(Simulator):
    import csdl
    from csdl import Model, GraphRepresentation
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.subtraction import SubtractionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.subtraction import SubtractionFunction
    from csdl.examples.models.hierarchical import Hierarchical
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.subtraction import SubtractionVectorFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    
    
    class ExampleParallelTargets(Model):
        # Use 1 input for several declared variables of the same instance
        # return sim[f_out] = 15
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            self.create_input('a', val=2.0)
            self.create_input('b', val=3.0)
    
            self.add(
                AdditionFunction(),
                name='addition1',
                promotes=['a', 'b'],
            )
    
            self.add(
                AdditionFunction(),
                name='addition2',
                promotes=['a', 'b'],
            )
    
            self.add(
                AdditionFunction(),
                name='addition3',
                promotes=['a', 'b'],
            )
    
            f1 = self.declare_variable('f1')
            f2 = self.declare_variable('f2')
            f3 = self.declare_variable('f3')
            self.connect('addition1.f', 'f1')
            self.connect('addition2.f', 'f2')
            self.connect('addition3.f', 'f3')
    
            self.register_output('f_out', f1 + f2 + f3)
    
    
    rep = GraphRepresentation(ExampleParallelTargets())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep