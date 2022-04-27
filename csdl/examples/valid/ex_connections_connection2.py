def example(Simulator):
    from csdl import Model
    import numpy as np
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.connect_within import ConnectWithin
    from csdl.examples.models.addition import AdditionFunction
    from csdl.examples.models.addition import AdditionFunction, ParallelAdditionFunction
    
    
    class ExampleConnection2(Model):
        # Connecting promoted variables
        # sim['y'] = 3
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import AdditionFunction
    
            a = self.create_input('a')
    
            self.add(AdditionFunction(), name='A')
    
            f1 = self.declare_variable('f1')
            self.register_output('y', a + f1)
            # Here we are connecting output f from model A to f1
            # f1 will have a value of 2.
            self.connect('f', 'f1')
    
    
    sim = Simulator(ExampleConnection2())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim