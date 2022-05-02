def example(Simulator):
    import csdl
    from csdl import Model, GraphRepresentation
    import numpy as np
    
    
    class ErrorScalarIncorrectOrder(Model):
        def define(self):
            scalar = self.declare_variable('scalar', val=1.)
            expanded_scalar = csdl.expand((2, 3), scalar)
            self.register_output('expanded_scalar', expanded_scalar)
    
    
    rep = GraphRepresentation(ErrorScalarIncorrectOrder())
    sim = Simulator(rep)
    sim.run()
    