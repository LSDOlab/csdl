def example(Simulator):
    from csdl import Model, GraphRepresentation, NonlinearBlockGS
    import csdl
    import numpy as np
    
    
    class ExampleUnary(Model):
        def define(self):
            x = self.declare_variable('x', val=np.pi)
            y = self.declare_variable('y', val=1)
            self.register_output('arccos', csdl.arccos(y))
            self.register_output('arcsin', csdl.arcsin(y))
            self.register_output('arctan', csdl.arctan(x))
            self.register_output('cos', csdl.cos(x))
            self.register_output('cosec', csdl.cosec(y))
            self.register_output('cosech', csdl.cosech(x))
            self.register_output('cosh', csdl.cosh(x))
            self.register_output('cotan', csdl.cotan(y))
            self.register_output('cotanh', csdl.cotanh(x))
            self.register_output('exp', csdl.exp(x))
            self.register_output('log', csdl.log(x))
            self.register_output('log10', csdl.log10(x))
            self.register_output('sec', csdl.sec(x))
            self.register_output('sech', csdl.sech(x))
            self.register_output('sin', csdl.sin(x))
            self.register_output('sinh', csdl.sinh(x))
            self.register_output('tan', csdl.tan(x))
            self.register_output('tanh', csdl.tanh(x))
    
    
    rep = GraphRepresentation(ExampleUnary())
    sim = Simulator(rep)
    sim.run()
    
    return sim, rep