from csdl import Model, GraphRepresentation
import csdl
import numpy as np

# TESTS:
# - Connection Example
# -- ExampleConnectDifferentLevels y=3.0
# -- ExampleConnectWithinSameModel y=5.0
# -- ExampleConnectPromotedOutputToDeclared y=3.0
# -- ExampleConnectInputToPromotedDeclared y=3.0
# -- ExampleNestedPromotedConnections y=8.0
# -- ExampleNestedUnpromotedConnections y=8.0
# -- ExampleNestedUnpromotedConnectionsVariation1 y=8.0
# -- ExampleNestedUnpromotedConnectionsVariation2 y=8.0
# -- ErrorConnectingWrongArgOrder
# -- ErrorConnectingTwoVariables
# -- ExampleConnectingVarsAcrossModels y=16.0
# -- ExampleConnectingVarsInModels
# -- ErrorConnectingCyclicalVars
# -- ExampleConnectingUnpromotedNames
# -- ErrorTwoWayConnection
# -- ExampleValueOverwriteConnection f=14.0
# -- ErrorConnectDifferentShapes
# -- ErrorConnectToNothing
# -- ExampleConnectCreateOutputs y=np.array([11,6])
# -- ErrorConnectCreateOutputs
# -- ExampleFalseCycle
# -- ExampleConnectUnpromotedNames y=3.0
# -- ExampleConnectUnpromotedNamesWithin y=3.0


class ExampleConnectDifferentLevels(Model):
    # Connecting Unromoted variables
    # sim['y'] = 3
    """
    :param var: y
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a')

        self.add(AdditionFunction(), name='A', promotes=[])

        f1 = self.declare_variable('f1')
        self.register_output('y', a + f1)
        self.connect('A.f', 'f1')


class ExampleConnectWithinSameModel(Model):
    # Connecting promoted variables
    # sim['y'] = 5
    """
    :param var: y
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a')
        b = self.create_input('b', val=2.0)
        c = a + b

        f1 = self.declare_variable('f1')
        self.register_output('y', c + f1)
        self.connect('b', 'f1')


class ExampleConnectPromotedOutputToDeclared(Model):
    # Connecting promoted variables
    # sim['y'] = 3
    """
    :param var: y
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a')

        self.add(AdditionFunction(), name='A')

        f1 = self.declare_variable('f1')
        self.register_output('y', a + f1)
        self.connect('f', 'f1')


class ExampleConnectInputToPromotedDeclared(Model):
    # Connecting promoted variables
    # sim['y'] = 3
    """
    :param var: y
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        x = self.create_input('x')

        self.add(AdditionFunction(), name='A')

        self.connect('x', 'a')


class ExampleNestedPromotedConnections(Model):
    # Connecting promoted variables
    # sim['y'] = 8
    """
    :param var: y
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
        self.register_output('y', a + b1)


class ExampleNestedUnpromotedConnections(Model):
    # Connecting unpromoted variables
    # sim['y'] = 8
    """
    :param var: y
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
        b4 = m3.declare_variable('b4_connect')
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
        self.register_output('y', a + b1)


class ExampleNestedUnpromotedConnectionsVariation1(Model):
    # Connecting unpromoted variables. 1 Model promoted
    # sim['y'] = 8
    """
    :param var: y
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
        b4 = m3.declare_variable('b4_connect')
        m3.register_output('b3', a3 + b4)

        m2.add(m3, name='m3', promotes=[])
        b3 = m2.declare_variable('b3_connect')
        m2.register_output('b2', a2 + b3)

        m1.add(m2, name='m2', promotes=[])
        b2 = m1.declare_variable('b2_connect')
        m1.register_output('b1', a1 + b2)

        self.add(m1, name='m1')
        b1 = self.declare_variable('b1_connect')

        self.connect('m1.b1', 'b1_connect')
        self.connect('m1.m2.b2', 'm1.b2_connect')
        self.connect('m1.m2.m3.b3', 'm1.m2.b3_connect')
        self.connect('m1.m2.m3.m4.b4', 'm1.m2.m3.b4_connect')
        self.register_output('y', a + b1)


class ExampleNestedUnpromotedConnectionsVariation2(Model):
    # Connecting unpromoted variables. 2 Models promoted.
    # Note: cannot connect variables in same model
    # sim['y'] = 8
    """
    :param var: y
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
        b4 = m3.declare_variable('b4_connect')
        m3.register_output('b3', a3 + b4)

        m2.add(m3, name='m3')
        b3 = m2.declare_variable('b3')
        m2.register_output('b2', a2 + b3)

        m1.add(m2, name='m2', promotes=[])
        b2 = m1.declare_variable('b2_connect')
        m1.register_output('b1', a1 + b2)

        self.add(m1, name='m1')
        b1 = self.declare_variable('b1_connect')

        # self.connect('b1', 'b1_connect')
        # self.connect('m2.b2', 'b2_connect')
        # self.connect('m2.m4.b4', 'm2.b4_connect')
        # m2.m4.b4
        self.connect('m1.b1', 'b1_connect')
        self.connect('m1.m2.b2', 'm1.b2_connect')
        self.connect('m1.m2.m3.b3', 'm1.m2.b3_connect')
        self.connect('m1.m2.m3.m4.b4', 'm1.m2.m3.b4_connect')

        self.register_output('y', a + b1)


class ErrorConnectingWrongArgOrder(Model):
    # self.connect argument order must be in correct order
    # return error

    def define(self):

        a = self.create_input('a')
        b1 = self.declare_variable('b1')

        self.connect('b1', 'a')
        self.register_output('y', a + b1)


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
    # return sim['y'] = 16
    """
    :param var: y
    """

    def define(self):

        a = self.create_input('a', val=3)

        m1 = Model()
        a1 = m1.declare_variable('a1')  # connect to a
        m1.register_output('b1', a1 + 3.0)
        self.add(m1)

        m2 = Model()
        a2 = m2.declare_variable('a2')  # connect to b1
        m2.register_output('b2', a2 + 3.0)
        self.add(m2)

        m3 = Model()
        a3 = m3.declare_variable('a3')  # connect to b1
        a4 = m3.declare_variable('a4')  # connect to b2
        m3.register_output('b3', a3 + a4)
        self.add(m3)

        b4 = self.declare_variable('b4')  # connect to b3
        self.register_output('y', b4 + 1)

        self.connect('a', 'a1')
        self.connect('b1', 'a2')
        self.connect('b1', 'a3')
        self.connect('b2', 'a4')
        self.connect('b3', 'b4')


class ExampleConnectingVarsInModels(Model):
    # Cannot make connections within models
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.connect_within import ConnectWithin

        self.add(ConnectWithin(
        ))  # Adding a connection within a model will throw an error


class ErrorConnectingCyclicalVars(Model):
    # Cannot make connections that make cycles
    # return error

    def define(self):

        a = self.create_input('a')

        model = Model()
        a = model.declare_variable('a')
        b = model.declare_variable('b', val=3.0)
        c = a * b
        model.register_output('y',
                              a + c)  # connect to b, creating a cycle
        self.add(model)

        self.connect('y', 'b')


class ExampleConnectingUnpromotedNames(Model):
    # Cannot make connections using unpromoted names
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        c = self.create_input('c', val=3)

        self.add(AdditionFunction(), name='m')

        self.connect('c', 'm.b')


class ErrorTwoWayConnection(Model):
    # Cannot make two connections between two variables
    # return error

    def define(self):

        model = Model()
        a = model.declare_variable(
            'a', val=2.0)  # connect to b, creating a cycle
        b = model.declare_variable(
            'b', val=2.0)  # connect to a, creating a cycle
        c = a * b
        model.register_output('y', a + c)

        self.add(model, promotes=[], name='model')
        self.connect('model.a', 'model.b')
        self.connect('model.b', 'model.a')


class ExampleValueOverwriteConnection(Model):
    # Connection should overwrite values
    # return sim['f'] = 14
    """
    :param var: y
    """

    def define(self):

        model = Model()
        a = model.declare_variable(
            'a', val=4.0)  # connect to b, creating a cycle
        b = model.declare_variable(
            'b', val=3.0)  # connect to a, creating a cycle
        c = a * b
        model.register_output('y', a + c)

        self.add(
            model,
            promotes=[],
            name='model',
        )

        d = self.declare_variable('d', val=10.0)
        self.register_output('f', 2 * d)
        self.connect('model.y', 'd')


class ErrorConnectDifferentShapes(Model):
    # Connections can't be made between two variables with different shapes
    # return error

    def define(self):

        d = self.create_input('d', shape=(2, 2))

        model = Model()
        a = model.declare_variable(
            'a', val=4.0)  # connect to b, creating a cycle
        b = model.declare_variable(
            'b', val=3.0)  # connect to a, creating a cycle
        c = a * b
        model.register_output('y', a + c)

        self.add(
            model,
            promotes=[],
            name='model',
        )

        y = self.declare_variable('model.y')
        self.register_output('f', y + 2.0)
        self.connect('d', 'model.a')


class ErrorConnectToNothing(Model):
    # Connections can't be made between variables not existing
    # return error

    def define(self):

        model = Model()
        a = model.create_input(
            'a', val=4.0)  # connect to b, creating a cycle
        b = model.declare_variable(
            'b', val=3.0)  # connect to a, creating a cycle
        c = a * b
        model.register_output('y', a + c)

        self.add(
            model,
            promotes=[],
            name='model',
        )

        a = self.declare_variable('model.a')
        self.register_output('f', a + 2.0)
        self.connect('model.a', 'c')


class ExampleConnectCreateOutputs(Model):
    # Connections should work for concatenations
    # return sim['y'] = [11, 6]
    """
    :param var: f
    """

    def define(self):

        a = self.create_input('a', val=5)

        b = self.declare_variable('b')
        c = self.create_output('c', shape=(2, ))
        c[0] = b + a
        c[1] = a

        d = self.declare_variable('d', shape=(2, ))
        self.register_output('y', d + np.ones((2, )))

        self.connect('a', 'b')  # Connecting a to b
        self.connect('c', 'd')  # Connecting a to b


class ErrorConnectCreateOutputs(Model):
    # Connections should not work for concatenations if wrong shape
    # return error

    def define(self):

        a = self.create_input('a', val=5)

        b = self.declare_variable('b')
        c = self.create_output('c', shape=(2, ))
        c[0] = b + a
        c[1] = a

        d = self.declare_variable('d')
        self.register_output('f', d + np.ones((2, )))

        self.connect('a', 'b')  # Connecting a to b
        self.connect('c', 'd')  # Connecting c to d but d is wrong shape


class ExampleFalseCycle(Model):
    # Adding models out of order and adding connections between them may create explicit relationships out of order
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.false_cycle import FalseCyclePost
        from csdl.examples.models.addition import AdditionFunction

        self.add(AdditionFunction())

        self.add(FalseCyclePost())

        self.connect('x', 'b')


class ExampleConnectUnpromotedNames(Model):
    # Connecting Unromoted variables
    # sim['y'] = 3
    """
    :param var: y
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a')

        self.add(AdditionFunction(), name='A')

        f1 = self.declare_variable('f1')
        self.register_output('y', a + f1)
        self.connect('A.f', 'f1')


class ExampleConnectUnpromotedNamesWithin(Model):
    # Connecting Unromoted variables
    # sim['y'] = 3
    """
    :param var: y
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        m = Model()  # model B
        a = m.create_input('a')
        b = m.create_input('b')
        m.register_output('f', a + b)

        mm = Model()  # model A
        c = mm.create_input('c')
        d = mm.declare_variable('d')
        mm.register_output('f1', c + d)
        mm.add(m, promotes=[], name='B')
        mm.connect('B.f', 'd')
        self.add(mm, 'A')

        f1 = self.declare_variable('f1')
        e = self.create_input('e')
        self.register_output('y', e + f1)

