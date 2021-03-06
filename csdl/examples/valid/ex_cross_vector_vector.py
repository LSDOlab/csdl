def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleVectorVector(Model):
        def define(self):
            # Creating two vectors
            vecval1 = np.arange(3)
            vecval2 = np.arange(3) + 1
    
            vec1 = self.declare_variable('vec1', val=vecval1)
            vec2 = self.declare_variable('vec2', val=vecval2)
    
            # Vector-Vector Cross Product
            self.register_output('VecVecCross', csdl.cross(vec1, vec2, axis=0))
    
    
    rep = GraphRepresentation(ExampleVectorVector())
    sim = Simulator(rep)
    sim.run()
    
    print('vec1', sim['vec1'].shape)
    print(sim['vec1'])
    print('vec2', sim['vec2'].shape)
    print(sim['vec2'])
    print('VecVecCross', sim['VecVecCross'].shape)
    print(sim['VecVecCross'])
    
    return sim, rep