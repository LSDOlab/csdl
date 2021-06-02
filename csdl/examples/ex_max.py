from csdl import Model
import csdl
import numpy as np


class ExampleScalar(Model):
    """
    :param var: tensor
    :param var: ScalarMin
    """
    def define(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6

        # Shape of a tensor
        tensor_shape = (m, n, o, p, q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input
        ten = self.declare_variable('tensor', val=val)

        # Computing the minimum across the entire tensor, returns single value
        self.register_output('ScalarMin', csdl.max(ten))


class ExampleAxiswise(Model):
    """
    :param var: tensor
    :param var: AxiswiseMin
    """
    def define(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6

        # Shape of a tensor
        tensor_shape = (m, n, o, p, q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input
        ten = self.declare_variable('tensor', val=val)

        # Computing the axiswise minimum on the tensor
        axis = 1
        self.register_output('AxiswiseMin', csdl.max(ten, axis=axis))


class ExampleElementwise(Model):
    """
    :param var: tensor1
    :param var: tensor2
    :param var: ElementwiseMin
    """
    def define(self):

        m = 2
        n = 3
        # Shape of the three tensors is (2,3)
        shape = (m, n)

        # Creating the values for two tensors
        val1 = np.array([[1, 5, -8], [10, -3, -5]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_variable('tensor1', val=val1)
        tensor2 = self.declare_variable('tensor2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMin', csdl.max(tensor1, tensor2))


class ErrorMultiInputsAndAxis(Model):
    def define(self):
        # Creating the values for two tensors
        val1 = np.array([[1, 5, -8], [10, -3, -5]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_variable('tensor1', val=val1)
        tensor2 = self.declare_variable('tensor2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMinWithAxis',
                             csdl.max(tensor1, tensor2, axis=0))


class ErrorInputsNotSameSize(Model):
    def define(self):
        # Creating the values for two tensors
        val1 = np.array([[1, 5], [10, -3]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_variable('tensor1', val=val1)
        tensor2 = self.declare_variable('tensor2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMinWrongSize',
                             csdl.max(tensor1, tensor2))
