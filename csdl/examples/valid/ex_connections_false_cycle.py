def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from connection_error import ConnectWithin
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.false_cycle import FalseCyclePost
    from csdl.examples.models.addition import AdditionFunction
    
    
    class ExampleFalseCycle(Model):
        # Adding models out of order and adding connections between them may create explicit relationships out of order
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.false_cycle import FalseCyclePost
            from csdl.examples.models.addition import AdditionFunction
    
            self.add(AdditionFunction())
    
            self.add(FalseCyclePost())
    
            self.connect('x', 'b')
    
    
    rep = GraphRepresentation(ExampleFalseCycle())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep