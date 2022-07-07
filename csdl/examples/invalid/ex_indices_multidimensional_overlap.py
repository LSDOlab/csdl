def example(Simulator):
    import numpy as np
    import csdl
    from csdl import Model, GraphRepresentation
    
    
    class ErrorMultidimensionalOverlap(Model):
        def define(self):
            z = self.declare_variable('z',
                                      shape=(2, 3),
                                      val=np.arange(6).reshape((2, 3)))
            x = self.create_output('x', shape=(2, 3))
            x[0:2, 0:3] = z
            # This triggers an error
            x[0:2, 0:3] = z
    
    
    rep = GraphRepresentation(ErrorMultidimensionalOverlap())
    sim = Simulator(rep)
    sim.run()
    