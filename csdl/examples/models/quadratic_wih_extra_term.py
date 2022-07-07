from csdl import Model, GraphRepresentation


class QuadraticWithExtraTerm(Model):
    def initialize(self):
        self.parameters.declare('shape', types=(tuple))

    def define(self):
        shape = self.parameters['shape']
        a = self.declare_variable('a', shape=shape)
        b = self.declare_variable('b', shape=shape)
        c = self.declare_variable('c', shape=shape)
        r = self.declare_variable('r', shape=shape)
        y = self.declare_variable('y', shape=shape)
        self.register_output('z', a * y**2 + b * y + c - r)
