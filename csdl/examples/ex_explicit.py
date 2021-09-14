from csdl import Model, NonlinearBlockGS
import csdl
import numpy as np


class ExampleLiterals(Model):
    """
    :param var: y
    """
    def define(self):
        x = self.declare_variable('x', val=3)
        y = -2 * x**2 + 4 * x + 3
        self.register_output('y', y)


class ExampleBinaryOperations(Model):
    """
    :param var: y1
    :param var: y2
    :param var: y3
    :param var: y4
    :param var: y5
    :param var: y6
    :param var: y7
    :param var: y8
    :param var: y9
    :param var: y10
    :param var: y11
    :param var: y12
    """
    def define(self):
        # declare inputs with default values
        x1 = self.declare_variable('x1', val=2)
        x2 = self.declare_variable('x2', val=3)
        x3 = self.declare_variable('x3', val=np.arange(7))

        # Expressions with multiple binary operations
        y1 = -2 * x1**2 + 4 * x2 + 3
        self.register_output('y1', y1)

        # Elementwise addition
        y2 = x2 + x1

        # Elementwise subtraction
        y3 = x2 - x1

        # Elementwise multitplication
        y4 = x1 * x2

        # Elementwise division
        y5 = x1 / x2
        y6 = x1 / 3
        y7 = 2 / x2

        # Elementwise Power
        y8 = x2**2
        y9 = x1**2

        self.register_output('y2', y2)
        self.register_output('y3', y3)
        self.register_output('y4', y4)
        self.register_output('y5', y5)
        self.register_output('y6', y6)
        self.register_output('y7', y7)
        self.register_output('y8', y8)
        self.register_output('y9', y9)

        # Adding other expressions
        self.register_output('y10', y1 + y7)

        # Array with scalar power
        y11 = x3**2
        self.register_output('y11', y11)

        # Array with array of powers
        y12 = x3**(2 * np.ones(7))
        self.register_output('y12', y12)


class ExampleCycles(Model):
    """
    :param var: cycle_1.x
    :param var: cycle_2.x
    :param var: cycle_3.x
    """
    def define(self):
        # x == (3 + x - 2 * x**2)**(1 / 4)
        model = Model()
        x = model.create_output('x')
        x.define((3 + x - 2 * x**2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add(model, name='cycle_1', promotes=[])

        # x == ((x + 3 - x**4) / 2)**(1 / 4)
        model = Model()
        x = model.create_output('x')
        x.define(((x + 3 - x**4) / 2)**(1 / 4))
        model.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add(model, name='cycle_2', promotes=[])

        # x == 0.5 * x
        model = Model()
        x = model.create_output('x')
        x.define(0.5 * x)
        model.nonlinear_solver = NonlinearBlockGS(maxiter=100)
        self.add(model, name='cycle_3', promotes=[])


class ExampleNoRegisteredOutput(Model):
    """
    :param var: prod
    """
    def define(self):
        model = Model()
        a = model.declare_variable('a', val=2)
        b = model.create_input('b', val=12)
        model.register_output('prod', a * b)
        self.add(model, name='sys')

        # These expressions are not passed to the compiler back end
        x1 = self.declare_variable('x1')
        x2 = self.declare_variable('x2')
        y1 = x2 + x1
        y2 = x2 - x1
        y3 = x1 * x2
        y5 = x2**2


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


class ExampleWithSubsystems(Model):
    """
    :param var: prod
    :param var: y1
    :param var: y2
    :param var: y3
    :param var: y4
    :param var: y5
    :param var: y6
    """
    def define(self):
        # Create input to main model
        x1 = self.create_input('x1', val=40)

        # Powers
        y4 = x1**2

        # Create subsystem that depends on previously created
        # input to main model
        m = Model()

        # This value is overwritten by connection from the main model
        a = m.declare_variable('x1', val=2)
        b = m.create_input('x2', val=12)
        m.register_output('prod', a * b)
        self.add(m, name='subsystem')

        # declare inputs with default values
        # This value is overwritten by connection
        # from the submodel
        x2 = self.declare_variable('x2', val=3)

        # Simple addition
        y1 = x2 + x1
        self.register_output('y1', y1)

        # Simple subtraction
        self.register_output('y2', x2 - x1)

        # Simple multitplication
        self.register_output('y3', x1 * x2)

        # Powers
        y5 = x2**2

        # register outputs in reverse order to how they are defined
        self.register_output('y5', y5)
        self.register_output('y6', y1 + y5)
        self.register_output('y4', y4)
