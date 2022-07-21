from csdl import Model, NonlinearBlockGS
import csdl
import numpy as np


class ExampleExponentialConstantScalarVariableScalar(Model):
    """
    :param var: y
    """
    def define(self):
        a = 5.0 
        x = self.declare_variable('x', val=3)
        y = csdl.exp_a(a, x)
        self.register_output('y', y)

class ExampleExponentialConstantScalarVariableArray(Model):
    """
    :param var: y
    """
    def define(self):
        a = 5.0
        x = self.declare_variable('x', val=np.array([1,2,3]))
        y = csdl.exp_a(a, x)
        self.register_output('y', y)

class ExampleExponentialConstantArrayVariableArray(Model):
    """
    :param var: y
    """
    def define(self):
        x = self.declare_variable('x', val=np.array([1,2,3]))
        a = np.array([1,2,3])
        y = csdl.exp_a(a, x)
        self.register_output('y', y)


class ErrorExponentialConstantArrayVariableScalar(Model):
    def define(self):
        x = self.declare_variable('x', val=2.0)
        a = np.array([1,2,3])
        y = csdl.exp_a(a, x)
        self.register_output('y', y) 