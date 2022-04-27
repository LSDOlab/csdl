def example(Simulator):
    from csdl import Model
    import numpy as np
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.connect_within import ConnectWithin
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionVectorFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.concatenate import ConcatenateFunction
    from csdl.examples.models.concatenate import ConcatenateFunction
    from csdl.examples.models.addition import ParallelAdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    
    
    class ExampleUnpromotedConnection(Model):
        # Connecting Unromoted variables
        # sim['y'] = 3
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            a = self.create_input('a')
    
            self.add(AdditionFunction(), name='A', promotes=[])
    
            f1 = self.declare_variable('f1')
            self.register_output('y', a + f1)
            # We can issue a connection between an unpromoted output f in model 'A'
            # and the declared variable 'f1'
            self.connect('A.f', 'f1')
    
    
    sim = Simulator(ExampleUnpromotedConnection())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim