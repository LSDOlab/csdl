def example(Simulator):
    import numpy as np
    import csdl
    from csdl import Model, GraphRepresentation
    
    
    class ErrorOneDimensionalReuse(Model):
        def define(self):
            n = 8
            u = self.declare_variable('u',
                                      shape=(n, ),
                                      val=np.arange(n).reshape((n, )))
            v = self.create_output('v', shape=(n, ))
            v[:4] = u[:4]
            v[4:] = u[:4]
    
    
    rep = GraphRepresentation(ErrorOneDimensionalReuse())
    sim = Simulator(rep)
    sim.run()
    