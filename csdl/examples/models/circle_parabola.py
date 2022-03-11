from csdl import Model


class CircleParabola(Model):
    def define(self):
        r = self.declare_variable('r')
        a = self.declare_variable('a')
        b = self.declare_variable('b')
        c = self.declare_variable('c')
        x = self.declare_variable('x', val=1.5)
        y = self.declare_variable('y', val=0.9)
        self.register_output('rx', x**2 + (y - r)**2 - r**2)
        self.register_output('ry', a * y**2 + b * y + c)


class CircleParabolaExpose(Model):
    def define(self):
        r = self.declare_variable('r')
        a = self.declare_variable('a')
        b = self.declare_variable('b')
        c = self.declare_variable('c')
        x = self.declare_variable('x', val=1.5)
        y = self.declare_variable('y', val=0.9)
        self.register_output('rx', x**2 + (y - r)**2 - r**2)
        self.register_output('ry', a * y**2 + b * y + c)
        self.register_output('t1', a + b + c)
        self.register_output('t2', x**2)
        self.register_output('t3', 2 * y)
        self.register_output('t4', x + y)
