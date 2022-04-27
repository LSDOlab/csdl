from csdl import Model
import numpy as np

# TESTS:
# - Connection Example
# -- ExampleUnpromotedConnection
# -- ExampleConnection1
# -- ExampleNestedPromotedConnections
# -- ExampleNestedUnpromotedConnections
# -- ExampleNestedUnpromotedConnectionsVariation1
# -- ExampleNestedUnpromotedConnectionsVariation2
# -- ErrorConnectingWrongArgOrder
# -- ErrorConnectingTwoVariables
# -- ExampleConnectingVarsAcrossModels
# -- ErrorConnectingVarsInModels
# -- ErrorConnectingCyclicalVars
# -- ErrorConnectingUnpromotedNames
# -- ErrorTwoWayConnection
# -- ExampleValueOverwriteConnection
# -- ErrorConnectDifferentShapes
# -- ErrorConnectVarToNothing
# -- ErrorConnectNothingToVar
# -- ErrorExistingConnections
# -- ExampleConnectCreateOutputs
# -- ErrorConnectCreateOutputs
# -- ErrorFalseCycle1
# -- ErrorFalseCycle2


class ExampleUnpromotedConnection(Model):
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
        # We can issue a connection between an unpromoted output f in model 'A'
        # and the declared variable 'f1'
        self.connect('A.f', 'f1')


class ExampleConnection1(Model):
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
        # Here we are connecting output f from model A to f1
        # f1 will have a value of 2.
        self.connect('f', 'f1')


class ExampleNestedPromotedConnections(Model):
    # Connecting promoted variables
    # sim['y'] = 8
    """
    :param var: y
    """

    def define(self):

        a = self.create_input('a')

        # Create 4 models each with 1 input and 1 output
        m1 = Model()
        a1 = m1.create_input('a1')

        m2 = Model()
        a2 = m2.create_input('a2')

        m3 = Model()
        a3 = m3.create_input('a3')

        m4 = Model()
        a4 = m4.create_input('a4')

        # Add the 4 models nested within eachother.
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

        # Issue connections between variables in each model from the parent model.
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

        # Create 4 models each with 1 input and 1 output
        m1 = Model()
        a1 = m1.create_input('a1')

        m2 = Model()
        a2 = m2.create_input('a2')

        m3 = Model()
        a3 = m3.create_input('a3')

        m4 = Model()
        a4 = m4.create_input('a4')

        # Add the 4 models nested within eachother.
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

        # Issue connections between variables in each model from the parent model.
        # All models are unpromoted so use unpromoted names.
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

        # Create 4 models each with 1 input and 1 output
        m1 = Model()
        a1 = m1.create_input('a1')

        m2 = Model()
        a2 = m2.create_input('a2')

        m3 = Model()
        a3 = m3.create_input('a3')

        m4 = Model()
        a4 = m4.create_input('a4')

        # Add the 4 models nested within eachother.
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

        # Issue connections between variables in each model from the parent model.
        # Be careful to use the correct unpromoted/promoted names
        self.connect('b1', 'b1_connect')
        self.connect('m2.b2', 'b2_connect')
        self.connect('m2.m3.b3', 'm2.b3_connect')
        self.connect('m2.m3.m4.b4', 'm2.m3.b4_connect')
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

        # Create 4 models each with 1 input and 1 output
        m1 = Model()
        a1 = m1.create_input('a1')

        m2 = Model()
        a2 = m2.create_input('a2')

        m3 = Model()
        a3 = m3.create_input('a3')

        m4 = Model()
        a4 = m4.create_input('a4')

        # Add the 4 models nested within eachother.
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

        # Issue connections between variables in each model from the parent model.
        # Be careful to use the correct unpromoted/promoted names.
        # Variable b3 is automatically promoted and connected.
        self.connect('b1', 'b1_connect')
        self.connect('m2.b2', 'b2_connect')
        self.connect('m2.m4.b4', 'm2.b4_connect')
        self.register_output('y', a + b1)


class ErrorConnectingWrongArgOrder(Model):
    # self.connect argument order must be in correct order
    # return error

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
        # Here we are connecting output f from model A to f1
        # f1 will have a value of 2.
        self.connect('f1', 'f')  # should be self.connect('f', 'f1')


class ErrorConnectingTwoVariables(Model):
    # Connecting two variables to same the variable returns error
    # return error
    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a1 = self.create_input('a1')
        a2 = self.create_input('a2')

        self.add(AdditionFunction(), name='A')

        f = self.declare_variable('f')
        self.register_output('y', a1 + a2 + f1)
        self.connect('a1', 'a')
        self.connect('a2', 'a')  # 'a1' is already connected to 'a' so we cannot connect 'a2' to 'a'


class ExampleConnectingVarsAcrossModels(Model):
    # Connecting variables accross multiple models
    # return sim['y'] = 16
    """
    :param var: y
    """

    def define(self):

        a = self.create_input('a', val=3)

        # Add three models sequentially
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
        self.register_output('y', b4+1)

        # We can connect variables between models sequentially
        self.connect('a', 'a1')
        self.connect('b1', 'a2')
        self.connect('b1', 'a3')
        self.connect('b2', 'a4')
        self.connect('b3', 'b4')


class ErrorConnectingVarsInModels(Model):
    # Cannot make connections within models
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.connect_within import ConnectWithin

        self.add(ConnectWithin())  # Adding a connection within a model will throw an error


class ErrorConnectingCyclicalVars(Model):
    # Cannot make connections that make cycles
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        a = self.create_input('a')

        self.add(AdditionFunction(), name='A')

        f = self.declare_variable('f')
        self.register_output('y', a + f)  # connect to b

        self.connect('y', 'b')  # connections creating a cycle will return an error


class ErrorConnectingUnpromotedNames(Model):
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

        self.connect('c', 'm.b')  # should be self.connect('c', 'b')


class ErrorTwoWayConnection(Model):
    # Cannot make two connections between two variables
    # return error

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

        # two directional connections cannot be done.
        self.connect('f', 'f1')
        self.connect('f1', 'f')


class ExampleValueOverwriteConnection(Model):
    # Connection should overwrite values
    # return sim['y'] = 3
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

        f1 = self.declare_variable('f1', val=10)
        self.register_output('y', a + f1)

        self.connect('f', 'f1')  # connect to 'f' to 'f1', value of 10 should be overwritten.


class ErrorConnectDifferentShapes(Model):
    # Connections can't be made between two variables with different shapes.
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionVectorFunction

        a = self.create_input('a', shape=(3,))

        self.add(AdditionVectorFunction(), name='A')

        f1 = self.declare_variable('f1')
        self.register_output('y', a + f1)

        self.connect('f', 'f1')  # Connecting variables with differing shapes will result in an error.


class ErrorConnectVarToNothing(Model):
    # Connections can't be made between variables not existing
    # return error

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
        # Here we are connecting output f from model A to f1
        # f1 will have a value of 2.
        self.connect('f', 'f2')  # Connecting non-existing variables will result in an error.


class ErrorConnectNothingToVar(Model):
    # Connections can't be made between variables not existing
    # return error

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

        self.connect('f2', 'f1')  # Connecting non-existing variables will result in an error.


class ErrorExistingConnections(Model):
    # Connections can't be made between variables that are already connected
    # return error

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
        self.connect('f', 'f1')  # Connecting already existing connections are not allowed.


class ExampleConnectCreateOutputs(Model):
    # Connections should work for concatenations
    # return sim['y'] = np.array([11. 6.])
    """
    :param var: y
    """

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.concatenate import ConcatenateFunction

        a = self.create_input('a', val=5)

        self.add(ConcatenateFunction())

        d = self.declare_variable('d', shape=(2,))
        self.register_output('y', d + np.ones((2,)))

        self.connect('a', 'b')
        self.connect('a', 'e')
        self.connect('c', 'd')  # We can issue connections from concatenations


class ErrorConnectCreateOutputs(Model):
    # Connections should not work for concatenations if wrong shape
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.concatenate import ConcatenateFunction

        a = self.create_input('a', val=5)

        self.add(ConcatenateFunction())

        d = self.declare_variable('d')
        self.register_output('f', d + np.ones((2,)))

        self.connect('a', 'b')
        self.connect('a', 'e')
        self.connect('c', 'd')  # We cannot issue connections from concatenations to a variable with the wrong shape


class ErrorFalseCycle1(Model):
    # Adding variables and connecting them in a certain way may create false cycles
    # ***NOT SURE IF ERROR OR NOT?***
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import ParallelAdditionFunction

        self.add(ParallelAdditionFunction(), name='model1', promotes=[])

        self.add(ParallelAdditionFunction(), name='model2', promotes=[])

        # This looks like a cycle but both model contain parallel computations:
        #    x_out = x_in + 1
        #    y_out = y_in + 1
        # However, these calculations can be done explicitly
        self.connect('model1.x_out', 'model2.x_in')
        self.connect('model2.y_out', 'model1.y_in')


class ErrorFalseCycle2(Model):
    # Adding variables and connecting them in a certain way may create false cycles
    # ***NOT SURE IF ERROR OR NOT?***
    # return error

    def define(self):
        # NOTE: Importing definitions within a method is bad practice.
        # This is only done here to automate example/test case
        # generation more easily.
        # When defining CSDL models, please put the import statements at
        # the top of your Python file(s).
        from csdl.examples.models.addition import AdditionFunction

        self.add(AdditionFunction(), name='model1', promotes=[])
        self.add(AdditionFunction(), name='model2', promotes=[])

        # This looks like a cycle because 'model2.f' is registered as an output after 'model1.f'.
        # However, these calculations can be done explicitly
        self.connect('model2.f', 'model1.a')
