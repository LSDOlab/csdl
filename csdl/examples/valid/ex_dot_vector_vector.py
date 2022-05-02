def example(Simulator):
    from csdl import Model, GraphRepresentation
    import csdl
    import numpy as np
    
    
    class ExampleVectorVector(Model):
        def define(self):
    
            m = 3
    
            # Shape of the vectors
            vec_shape = (m, )
    
            # Values for the two vectors
            vec1 = np.arange(m)
            vec2 = np.arange(m, 2 * m)
    
            # Adding the vectors to csdl
            vec1 = self.declare_variable('vec1', val=vec1)
            vec2 = self.declare_variable('vec2', val=vec2)
    
            # Vector-Vector Dot Product
            self.register_output('VecVecDot', csdl.dot(vec1, vec2))
    
    
    rep = GraphRepresentation(ExampleVectorVector())
    sim = Simulator(rep)
    sim.run()
    
    print('vec1', sim['vec1'].shape)
    print(sim['vec1'])
    print('vec2', sim['vec2'].shape)
    print(sim['vec2'])
    print('VecVecDot', sim['VecVecDot'].shape)
    print(sim['VecVecDot'])
    
    return sim, rep