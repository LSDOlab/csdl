def example(Simulator):
    import numpy as np
    import csdl
    from csdl import Model, GraphRepresentation
    
    
    class ErrorOneDimensionalOutOfRange(Model):
        def define(self):
            n = 20
            x = self.declare_variable('x',
                                      shape=(n - 4, ),
                                      val=np.arange(n - 4).reshape(
                                          (n - 4, )))
            y = self.declare_variable('y',
                                      shape=(4, ),
                                      val=16 + np.arange(4).reshape((4, )))
            z = self.create_output('z', shape=(n, ))
            z[0:n - 4] = 2 * (x + 1)
            # This triggers an error
            z[n - 3:n + 1] = y - 3
    
    
    rep = GraphRepresentation(ErrorOneDimensionalOutOfRange())
    sim = Simulator(rep)
    sim.run()
    