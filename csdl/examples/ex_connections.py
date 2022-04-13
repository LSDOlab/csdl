from csdl import Model
import csdl
import numpy as np


class ExampleUnpromotedConnection(Model):
    # Connecting Unpromoted variables
    # sim['f'] = 5
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a')

        m = Model()
        a1 = m.create_input('a1')
        m.register_output('b1', a1 + 3)

        self.add(m, name='A', promotes=[])

        b2 = self.declare_variable('b2')
        self.register_output('f', a + b2)
        self.connect('A.b1', 'b2')


class ExampleConnection(Model):
    # Connecting promoted variables
    # sim['f'] = 5
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a')

        m = Model()
        a1 = m.create_input('a1')
        m.register_output('b1', a1 + 3)

        self.add(m, name='A')

        b2 = self.declare_variable('b2')
        self.register_output('f', a + b2)
        self.connect('b1', 'b2')


class ExampleNestedPromotedConnections(Model):
    # Connecting promoted variables
    # sim['f'] = 8
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a')

        m1 = Model()
        a1 = m1.create_input('a1')

        m2 = Model()
        a2 = m2.create_input('a2')

        m3 = Model()
        a3 = m3.create_input('a3')

        m4 = Model()
        a4 = m4.create_input('a4')

        m4.register_output('b4', a4 + 3)

        m3.add(m4)
        b4 = m2.declare_variable('b4_connect')
        m3.register_output('b3', a3 + b4)

        m2.add(m3)
        b3 = m2.declare_variable('b3_connect')
        m2.register_output('b2', a2 + b3)

        m1.add(m2)
        b2 = m1.declare_variable('b2_connect')
        m1.register_output('b1', a1 + b2)

        self.add(m1)
        b1 = self.declare_variable('b1_connect')

        self.connect('b1', 'b1_connect')
        self.connect('b2', 'b2_connect')
        self.connect('b3', 'b3_connect')
        self.connect('b4', 'b4_connect')
        self.register_output('f', a + b1)


class ExampleNestedUnpromotedConnections(Model):
    # Connecting unpromoted variables
    # sim['f'] = 8
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a')

        m1 = Model()
        a1 = m1.create_input('a1')

        m2 = Model()
        a2 = m2.create_input('a2')

        m3 = Model()
        a3 = m3.create_input('a3')

        m4 = Model()
        a4 = m4.create_input('a4')

        m4.register_output('b4', a4 + 3)

        m3.add(m4, name='m4', promotes=[])
        b4 = m2.declare_variable('b4_connect')
        m3.register_output('b3', a3 + b4)

        m2.add(m3, name='m3', promotes=[])
        b3 = m2.declare_variable('b3_connect')
        m2.register_output('b2', a2 + b3)

        m1.add(m2, name='m2', promotes=[])
        b2 = m1.declare_variable('b2_connect')
        m1.register_output('b1', a1 + b2)

        self.add(m1, name='m1', promotes=[])
        b1 = self.declare_variable('b1_connect')

        self.connect('m1.b1', 'b1_connect')
        self.connect('m1.m2.b2', 'm1.b2_connect')
        self.connect('m1.m2.m3.b3', 'm1.m2.b3_connect')
        self.connect('m1.m2.m3.m4.b4', 'm1.m2.m3.b4_connect')
        self.register_output('f', a + b1)


class ExampleNestedUnpromotedConnectionsVariation1(Model):
    # Connecting unpromoted variables. 1 Model promoted
    # sim['f'] = 8
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a')

        m1 = Model()
        a1 = m1.create_input('a1')

        m2 = Model()
        a2 = m2.create_input('a2')

        m3 = Model()
        a3 = m3.create_input('a3')

        m4 = Model()
        a4 = m4.create_input('a4')

        m4.register_output('b4', a4 + 3)

        m3.add(m4, name='m4', promotes=[])
        b4 = m2.declare_variable('b4_connect')
        m3.register_output('b3', a3 + b4)

        m2.add(m3, name='m3', promotes=[])
        b3 = m2.declare_variable('b3_connect')
        m2.register_output('b2', a2 + b3)

        m1.add(m2, name='m2', promotes=[])
        b2 = m1.declare_variable('b2_connect')
        m1.register_output('b1', a1 + b2)

        self.add(m1, name='m1')
        b1 = self.declare_variable('b1_connect')

        self.connect('b1', 'b1_connect')
        self.connect('m2.b2', 'b2_connect')
        self.connect('m2.m3.b3', 'm2.b3_connect')
        self.connect('m2.m3.m4.b4', 'm2.m3.b4_connect')
        self.register_output('f', a + b1)


class ExampleNestedUnpromotedConnectionsVariation2(Model):
    # Connecting unpromoted variables. 2 Models promoted.
    # Note: cannot connect variables in same model
    # sim['f'] = 8
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a')

        m1 = Model()
        a1 = m1.create_input('a1')

        m2 = Model()
        a2 = m2.create_input('a2')

        m3 = Model()
        a3 = m3.create_input('a3')

        m4 = Model()
        a4 = m4.create_input('a4')

        m4.register_output('b4', a4 + 3)

        m3.add(m4, name='m4', promotes=[])
        b4 = m2.declare_variable('b4_connect')
        m3.register_output('b3', a3 + b4)

        m2.add(m3, name='m3')
        b3 = m2.declare_variable('b3')
        m2.register_output('b2', a2 + b3)

        m1.add(m2, name='m2', promotes=[])
        b2 = m1.declare_variable('b2_connect')
        m1.register_output('b1', a1 + b2)

        self.add(m1, name='m1')
        b1 = self.declare_variable('b1_connect')

        self.connect('b1', 'b1_connect')
        self.connect('m2.b2', 'b2_connect')
        self.connect('m2.m4.b4', 'm2.b4_connect')
        self.register_output('f', a + b1)


class ErrorConnectingWrongArgOrder(Model):
    # self.connect argument order must be in correct order
    # return error

    def define(self):

        a = self.create_input('a')
        b1 = self.declare_variable('b1')

        self.connect('b1', 'a')
        self.register_output('f', a + b1)


class ErrorConnectingTwoVariables(Model):
    # Connecting two variables to same the variable returns error
    # return error

    def define(self):

        a = self.create_input('a', val=3)
        b = self.declare_variable('b')

        self.register_output('f', a + b)

        c = self.declare_variable('c')
        d = self.declare_variable('d')
        self.register_output('f2', c + d)

        self.connect('f', 'c')
        self.connect('a', 'c')


class ExampleConnectingVarsAcrossModels(Model):
    # Connecting variables accross multiple models
    # return sim['f'] = 16
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a', val=3)

        m1 = Model()
        a1 = m1.declare_variable('a1')  # connect to a
        m1.register_output('b1', a1+3.0)
        self.add(m1)

        m2 = Model()
        a2 = m2.declare_variable('a2')  # connect to b1
        m2.register_output('b2', a2+3.0)
        self.add(m2)

        m3 = Model()
        a3 = m3.declare_variable('a3')  # connect to b1
        a4 = m3.declare_variable('a4')  # connect to b2
        m3.register_output('b3', a3+a4)
        self.add(m3)

        b4 = self.declare_variable('b4')  # connect to b3
        self.register_output('f', b4+1)

        self.connect('a', 'a1')
        self.connect('b1', 'a2')
        self.connect('b1', 'a3')
        self.connect('b2', 'a4')
        self.connect('b3', 'b4')


class ErrorConnectingVarsInModels(Model):
    # Cannot make connections within models
    # return error

    def define(self):

        a = self.create_input('a', val=3)

        m = Model()
        a1 = m.declare_variable('a')
        a2 = m.declare_variable('a2')  # connect to a

        m.register_output('f', a1+a2)
        m.connect('a', 'a2')

        self.add(m)


class ErrorConnectingCyclicalVars(Model):
    # Cannot make connections that make cycles
    # return error

    def define(self):

        a = self.create_input('a', val=3)
        b = self.declare_variable('b')

        c = a*b

        self.register_output('f', a+c)  # connect to b, creating a cycle

        self.connect('f', 'b')


class ErrorConnectingUnpromotedNames(Model):
    # Cannot make connections using unpromoted names
    # return error

    def define(self):

        a = self.create_input('a', val=3)

        m = Model()
        b = m.declare_variable('b')  # connect to a
        m.register_output('f', b+2)
        self.add(m, name='m')

        self.connect('a', 'm.b')


class ErrorTwoWayConnection(Model):
    # Cannot make two connections between two variables
    # return error

    def define(self):

        a = self.create_input('a', val=3)  # connect to b, creating a cycle
        b = self.declare_variable('b')  # connect to a, creating a cycle

        c = a*b

        self.register_output('f', a+c)

        self.connect('a', 'b')
        self.connect('b', 'a')


class ExampleValueOverwriteConnection(Model):
    # Connection should overwrite values
    # return sim[f] = 6
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a', val=3)
        b = self.declare_variable('b', val=10)  # connect to a, value of 10 should be overwritten

        self.register_output('f', a + b)

        self.connect('a', 'b')


class ExampleConnectionIgnore(Model):
    # Connections should make 'ignored' variables 'unignored'
    # ****** NOT SURE IF THIS SHOULD RETURN AN ERROR OR NOT ******
    # return sim['f'] = 15
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a', val=3)
        b = self.create_input('b', val=10)
        c_connect = b+2

        c = self.declare_variable('c')
        self.register_output('f', a + c)

        self.connect(c_connect.name, 'c')  # Not sure if this should throw an error or not...


class ErrorConnectDifferentShapes(Model):
    # Connections can't be made between two variables with different shapes
    # return error

    def define(self):

        a = self.create_input('a', val=np.array([3, 3]))
        b = self.declare_variable('b', shape=(1,))

        self.register_output('f', b + 2.0)

        self.connect('a', 'b')  # Connecting wrong shapes should be an error


class ErrorConnectToNothing(Model):
    # Connections can't be made between variables not existing
    # return error

    def define(self):

        a = self.create_input('a')

        self.register_output('f', a + 2.0)

        self.connect('a', 'b')  # Connecting to a non-existing variable is an error


class ExampleConnectCreateOutputs(Model):
    # Connections should work for concatenations
    # return sim['f'] = [11, 6]
    """
    :param var: f
    """
    def define(self):

        a = self.create_input('a', val=5)

        b = self.declare_variable('b')
        c = self.create_output('c', shape=(2,))
        c[0] = b + a
        c[1] = a

        d = self.declare_variable('d', shape=(2,))
        self.register_output('f', d + np.ones((2,)))

        self.connect('a', 'b')  # Connecting a to b
        self.connect('c', 'd')  # Connecting a to b


class ErrorConnectCreateOutputs(Model):
    # Connections should not work for concatenations if wrong shape
    # return error

    def define(self):

        a = self.create_input('a', val=5)

        b = self.declare_variable('b')
        c = self.create_output('c', shape=(2,))
        c[0] = b + a
        c[1] = a

        d = self.declare_variable('d')
        self.register_output('f', d + np.ones((2,)))

        self.connect('a', 'b')  # Connecting a to b
        self.connect('c', 'd')  # Connecting c to d but d is wrong shape
