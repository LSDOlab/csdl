from csdl import Model


class QuadraticFunction(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']
        a = self.declare_variable('a', shape=shape) # 1
        b = self.declare_variable('b', shape=shape) # -4 
        c = self.declare_variable('c', shape=shape) # 3
        x = self.declare_variable('x', shape=shape)
        y = a * x**2 + b * x + c
        self.register_output('y', y)


class QuadraticFunctionExpose(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)

    def define(self):
        shape = self.parameters['shape']
        a = self.declare_variable('a', shape=shape)
        b = self.declare_variable('b', shape=shape)
        c = self.declare_variable('c', shape=shape)
        x = self.declare_variable('x', shape=shape)
        r = self.declare_variable('r', shape=shape, val=0)
        y = a * x**2 + b * x + c - r
        self.register_output('y', y)
        self.register_output('t', a + b + c)
        self.register_output('t3', a + b + c - r)
        self.register_output('t4', y**2)
