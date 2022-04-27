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
    
    
    class ErrorConnectCreateOutputs(Model):
        # Connections should not work for concatenations if wrong shape
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.concatenate import ConcatenateFunction
    
            a = self.create_input('a', val=5)
    
            self.add(ConcatenateFunction())
    
            d = self.declare_variable('d')
            self.register_output('f', d + np.ones((2,)))
    
            self.connect('a', 'b')
            self.connect('a', 'e')
            self.connect('c', 'd')  # We cannot issue connections from concatenations to a variable with the wrong shape
    
    
    sim = Simulator(ErrorConnectCreateOutputs())
    sim.run()
    