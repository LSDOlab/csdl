from csdl import Model, GraphRepresentation
import csdl
import numpy as np


class ExampleAxisFree(Model):
    """
    :param var: in1
    :param var: axis_free_pnorm
    """
    def define(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_variable('in1', val=tensor)

        # Computing the 6-norm on in1 without specifying an axis
        self.register_output('axis_free_pnorm', csdl.pnorm(in1, pnorm_type=6))


class ExampleAxisWise(Model):
    """
    :param var: in1
    :param var: axiswise_pnorm
    """
    def define(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_variable('in1', val=tensor)

        # Computing the 6-norm of in1 over the specified axes.
        self.register_output('axiswise_pnorm',
                             csdl.pnorm(in1, axis=(1, 3), pnorm_type=6))


class ErrorTypeNotPositive(Model):
    def define(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_variable('in1', val=tensor)

        # Computing the 6-norm on in1 without specifying an axis
        self.register_output('axis_free_pnorm', csdl.pnorm(in1, pnorm_type=-2))


class ErrorTypeNotEven(Model):
    def define(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_variable('in1', val=tensor)

        # Computing the 6-norm on in1 without specifying an axis
        self.register_output('axis_free_pnorm', csdl.pnorm(in1, pnorm_type=3))
