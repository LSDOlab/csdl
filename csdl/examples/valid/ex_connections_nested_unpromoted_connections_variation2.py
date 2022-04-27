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
    
    
    class ExampleNestedUnpromotedConnectionsVariation2(Model):
        # Connecting unpromoted variables. 2 Models promoted.
        # Note: cannot connect variables in same model
        # sim['y'] = 8
    
        def define(self):
    
            a = self.create_input('a')
    
            # Create 4 models each with 1 input and 1 output
            m1 = Model()
            a1 = m1.create_input('a1')
    
            m2 = Model()
            a2 = m2.create_input('a2')
    
            m3 = Model()
            a3 = m3.create_input('a3')
    
            m4 = Model()
            a4 = m4.create_input('a4')
    
            # Add the 4 models nested within eachother.
            m4.register_output('b4', a4 + 3)
    
            m3.add(m4, name='m4', promotes=[])
            b4 = m2.declare_variable('b4_connect')
            m3.register_output('b3', a3 + b4)
    
            m2.add(m3, name='m3')
            b3 = m2.declare_variable('b3')
            m2.register_output('b2', a2 + b3)
    
            m1.add(m2, name='m2', promotes=[])
            b2 = m1.declare_variable('b2_connect')
            m1.register_output('b1', a1 + b2)
    
            self.add(m1, name='m1')
            b1 = self.declare_variable('b1_connect')
    
            # Issue connections between variables in each model from the parent model.
            # Be careful to use the correct unpromoted/promoted names.
            # Variable b3 is automatically promoted and connected.
            self.connect('b1', 'b1_connect')
            self.connect('m2.b2', 'b2_connect')
            self.connect('m2.m4.b4', 'm2.b4_connect')
            self.register_output('y', a + b1)
    
    
    sim = Simulator(ExampleNestedUnpromotedConnectionsVariation2())
    sim.run()
    
    print('y', sim['y'].shape)
    print(sim['y'])
    
    return sim