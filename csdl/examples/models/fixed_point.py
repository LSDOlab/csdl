from csdl import Model


class FixedPoint1(Model):
    def initialize(self):
        self.parameters.declare('name', types=str)

    def define(self):
        x = self.declare_variable(self.parameters['name'])
        r = self.register_output('r', x - (3 + x - 2 * x**2)**(1 / 4))


class FixedPoint2(Model):
    def initialize(self):
        self.parameters.declare('name', types=str)

    def define(self):
        x = self.declare_variable(self.parameters['name'])
        r = self.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))


class FixedPoint3(Model):
    def initialize(self):
        self.parameters.declare('name', types=str)

    def define(self):

        x = self.declare_variable(self.parameters['name'])
        r = self.register_output('r', x - 0.5 * x)


class FixedPoint1Expose(Model):
    def initialize(self):
        self.parameters.declare('name', types=str)

    def define(self):
        x = self.declare_variable(self.parameters['name'])
        r = self.register_output('r', x - (3 + x - 2 * x**2)**(1 / 4))
        self.register_output('t1', x**2)


class FixedPoint2Expose(Model):
    def initialize(self):
        self.parameters.declare('name', types=str)

    def define(self):
        x = self.declare_variable(self.parameters['name'])
        r = self.register_output('r', x - ((x + 3 - x**4) / 2)**(1 / 4))
        self.register_output('t2', x**2)


class FixedPoint3Expose(Model):
    def initialize(self):
        self.parameters.declare('name', types=str)

    def define(self):

        x = self.declare_variable(self.parameters['name'])
        r = self.register_output('r', x - 0.5 * x)
