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
    
    
    class ErrorFalseCycle1(Model):
        # Adding models out of order and adding connections between them may create explicit relationships out of order
        # ***NOT SURE IF ERROR OR NOT?***
        # return error
    
        def define(self):
            # NOTE: Importing definitions within a method is bad practice.
            # This is only done here to automate example/test case
            # generation more easily.
            # When defining CSDL models, please put the import statements at
            # the top of your Python file(s).
            from csdl.examples.models.addition import ParallelAdditionFunction
    
            self.add(ParallelAdditionFunction(), name='model1', promotes=[])
    
            self.add(ParallelAdditionFunction(), name='model2', promotes=[])
    
            # This looks like a cycle but both model contain parallel computations:
            #    x_out = x_in + 1
            #    y_out = y_in + 1
            # However, these calculations can be done explicitly
            self.connect('model1.x_out', 'model2.x_in')
            self.connect('model2.y_out', 'model1.y_in')
    
    
    sim = Simulator(ErrorFalseCycle1())
    sim.run()
    