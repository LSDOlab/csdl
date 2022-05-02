import csdl
from csdl import Model, GraphRepresentation
import numpy as np


class ExampleScalar2Array(Model):
    """
    :param var: scalar
    :param var: expanded_scalar
    """
    def define(self):
        # Expanding a scalar into an array
        scalar = self.declare_variable('scalar', val=1.)
        expanded_scalar = csdl.expand(scalar, (2, 3))
        self.register_output('expanded_scalar', expanded_scalar)


class ExampleArray2HigherArray(Model):
    """
    :param var: array
    :param var: expanded_array
    """
    def define(self):
        # Expanding an array into a higher-rank array
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_variable('array', val=val)
        expanded_array = csdl.expand(array, (2, 4, 3, 1), 'ij->iajb')
        self.register_output('expanded_array', expanded_array)


class ErrorScalarIncorrectOrder(Model):
    def define(self):
        scalar = self.declare_variable('scalar', val=1.)
        expanded_scalar = csdl.expand((2, 3), scalar)
        self.register_output('expanded_scalar', expanded_scalar)


class ErrorArrayNoIndices(Model):
    def define(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_variable('array', val=val)
        expanded_array = csdl.expand(array, (2, 4, 3, 1))
        self.register_output('expanded_array', expanded_array)


class ErrorArrayInvalidIndices1(Model):
    def define(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_variable('array', val=val)
        expanded_array = csdl.expand(array, (2, 4, 3, 1), 'ij->iaj')
        self.register_output('expanded_array', expanded_array)


class ErrorArrayInvalidIndices2(Model):
    def define(self):
        # Test array expansion
        val = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        array = self.declare_variable('array', val=val)
        expanded_array = csdl.expand(array, (2, 4, 3, 1), 'ij->ijab')
        self.register_output('expanded_array', expanded_array)
