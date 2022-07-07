from csdl import Model, GraphRepresentation
import numpy as np


class Hierarchical(Model):

    def define(self):

        x0 = self.create_input('x0', val=3.0)

        self.add(
            ModelB(),
            promotes=['x0', 'x1', 'x4'],
            name='ModelB',
        )

        self.add(
            ModelF(),
            promotes=[],
            name='ModelF',
        )

        x3 = self.declare_variable('x3')
        self.register_output('x3_out', x3 * 1.0)  # should be 1.0

        x4 = self.declare_variable('x4')
        self.register_output('x4_out', x4 * 1.0)  # should be 10.0


class ModelB(Model):

    def define(self):

        x0 = self.declare_variable('x0')

        self.add(
            ModelC(),
            promotes=['x0', 'x4'],
            name='ModelC',
        )

        self.add(
            ModelE(),
            promotes=['x1'],
            name='ModelE',
        )

        x1 = self.declare_variable('x1')

        self.register_output('x2',
                             x0 + x1)  # should be 3.0 + 3.141 = 6.141


class ModelC(Model):

    def define(self):

        x0 = self.declare_variable('x0')

        self.add(
            ModelD(),
            promotes=['x2', 'x4'],
            name='ModelD',
        )

        self.register_output('x1', x0 + 2)  # should be 3.0 + 2 = 5.0


class ModelD(Model):

    def define(self):

        x5 = self.create_input('x5', val=7.0)

        self.register_output('x2', x5 + 2)  # should be 9.0

        self.register_output('x4', x5 + 3)


class ModelE(Model):

    def define(self):

        x1 = self.create_input('x1', val=3.141)


class ModelF(Model):

    def define(self):

        x0 = self.declare_variable('x0')

        self.add(
            ModelG(),
            promotes=['x3'],
            name='ModelG',
        )

        self.add(
            ModelH(),
            promotes=['x3'],
            name='ModelH',
        )


class ModelG(Model):

    def define(self):

        x3 = self.create_input('x3', val=0.01)


class ModelH(Model):

    def define(self):

        x3 = self.declare_variable('x3')

        self.register_output('x3_out', x3 * 1.0)  # should be 0.01
