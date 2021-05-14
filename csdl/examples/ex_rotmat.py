from csdl import Model
import csdl
import numpy as np


class ExampleScalarRotX(Model):
    """
    :param var: scalar
    :param var: scalar_Rot_x
    """
    def define(self):
        angle_val3 = np.pi / 3

        angle_scalar = self.declare_variable('scalar', val=angle_val3)

        # Rotation in the x-axis for scalar
        self.register_output('scalar_Rot_x', csdl.rotmat(angle_scalar,
                                                         axis='x'))


class ExampleScalarRotY(Model):
    """
    :param var: scalar
    :param var: scalar_Rot_y
    """
    def define(self):
        angle_val3 = np.pi / 3

        angle_scalar = self.declare_variable('scalar', val=angle_val3)

        # Rotation in the y-axis for scalar
        self.register_output('scalar_Rot_y', csdl.rotmat(angle_scalar,
                                                         axis='y'))


class ExampleSameRadianTensorRotX(Model):
    """
    :param var: tensor
    :param var: tensor_Rot_x
    """
    def define(self):

        # Shape of a random tensor rotation matrix
        shape = (2, 3, 4)

        num_elements = np.prod(shape)

        # Tensor of angles in radians
        angle_val1 = np.repeat(np.pi / 3, num_elements).reshape(shape)

        # Adding the tensor as an input
        angle_tensor1 = self.declare_variable('tensor', val=angle_val1)

        # Rotation in the x-axis for tensor1
        self.register_output('tensor_Rot_x',
                             csdl.rotmat(angle_tensor1, axis='x'))


class ExampleDiffRadianTensorRotX(Model):
    """
    :param var: tensor
    :param var: tensor_Rot_x
    """
    def define(self):

        # Shape of a random tensor rotation matrix
        shape = (2, 3, 4)

        num_elements = np.prod(shape)

        # Vector of angles in radians
        angle_val2 = np.repeat(
            np.pi / 3, num_elements) + 2 * np.pi * np.arange(num_elements)

        angle_val2 = angle_val2.reshape(shape)

        # Adding the vector as an input
        angle_tensor = self.declare_variable('tensor', val=angle_val2)

        # Rotation in the x-axis for tensor2
        self.register_output('tensor_Rot_x', csdl.rotmat(angle_tensor,
                                                         axis='x'))
