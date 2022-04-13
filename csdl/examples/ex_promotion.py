from csdl import Model
import csdl
import numpy as np


class ErrorSameInputsPromoted(Model):
    # Can't create two inputs in two models with same names
    # Return error

    def define(self):

        a1 = self.create_input('a')

        m = Model()
        a2 = m.create_input('a')
        m.register_output('f1', a2 + 1)

        self.register_output('f2', a1 + 1)

        self.add(m)


class ErrorSameOutputsPromoted(Model):
    # Can't create two outputs in two models with same names
    # Return error

    def define(self):

        a1 = self.create_input('a1')

        m = Model()
        a2 = m.create_input('a2')
        m.register_output('f', a2 + 1)

        self.add(m)
        self.register_output('f', a1 + 1)


class ExampleSameIOUnpromoted(Model):
    # Can create two outputs in two models with same names if unpromoted
    #  sim['b'] should be = 2

    def define(self):

        a1 = self.create_input('a1')

        m = Model()
        a2 = m.create_input('a2')
        m.register_output('f', a2 + 1)

        self.add(m, promotes=['a2'])
        self.register_output('f', a1 + 1)


class ErrorTwoModelsPromoted(Model):
    # Can't have two models with same variable names if both promoted
    # Return error

    def define(self):

        m1 = Model()
        a1 = m1.create_input('a1')
        m1.register_output('f', a1 + 1)

        m2 = Model()
        a2 = m2.create_input('a1')
        m2.register_output('f', a2 + 1)
        self.add(m1)
        self.add(m2)


class ExampleTwoModelsUnpromoted(Model):
    # Can have two models with same variable names if  unpromoted
    # Return no error

    def define(self):

        m1 = Model()
        a1 = m1.create_input('a1')
        m1.register_output('f', a1 + 1)

        m2 = Model()
        a2 = m2.create_input('a1')
        m2.register_output('f', a2 + 1)
        self.add(m1)
        self.add(m2, promotes=[])


class ExampleUnconnectedVars(Model):
    # Declared variable with same local name should not be connected if unpromoted
    # sim['b'] should be = 2
    """
    :param var: b
    """

    def define(self):

        m = Model()
        a = m.create_input('a', val=3.0)
        m.register_output('f', a + 1)

        self.add(m, promotes=[])

        a1 = self.declare_variable('a')

        self.register_output('b', a1+1.0)


class ErrorStackedModels(Model):
    # Promotions should work for models within models
    # return error

    def define(self):

        a = self.create_input('a', val=3.0)

        m = Model()
        am = m.create_input('am', val=2.0)
        mm = Model()
        amm = mm.create_input('a', val=1.0)
        mm.register_output('bmm', amm*2.0)
        m.add(mm)
        bmm = m.declare_variable('bmm')
        m.register_output('bm', bmm+am)
        self.add(m)

        bm = self.declare_variable('bm')
        self.register_output('b', bm+a)


class ExampleWrongShape(csdl.Model):
    # Promotions should not be made if two variables with different shapes
    # return sim['f'] = 4

    def define(self):
        import numpy as np
        a = self.create_input('a', val=3.0)

        m = csdl.Model()
        am = m.create_input('am', val=np.array([2.0, 3.0]))
        m.register_output('bm', am+np.array([2.0, 3.0]))
        self.add(m)  # should not auto promote as it would create namespace errors (?) with 'bm'

        bm = self.declare_variable('bm')
        self.register_output('f', bm+a)
