from csdl import Model
import csdl
import numpy as np


class ExampleVectorVector(Model):
    """
    :param var: vec1
    :param var: vec2
    :param var: VecVecCross
    """
    def define(self):
        # Creating two vectors
        vecval1 = np.arange(3)
        vecval2 = np.arange(3) + 1

        vec1 = self.declare_variable('vec1', val=vecval1)
        vec2 = self.declare_variable('vec2', val=vecval2)

        # Vector-Vector Cross Product
        self.register_output('VecVecCross', csdl.cross(vec1, vec2, axis=0))


class ExampleTensorTensor(Model):
    """
    :param var: ten1
    :param var: ten2
    :param var: TenTenCross
    """
    def define(self):
        # Creating two tensors
        shape = (2, 5, 4, 3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        ten1 = self.declare_variable('ten1', val=tenval1)
        ten2 = self.declare_variable('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', csdl.cross(ten1, ten2, axis=3))


class ErrorDifferentShapes(Model):
    def define(self):
        # Creating two tensors
        shape1 = (2, 5, 4, 3)
        shape2 = (7, 5, 6, 3)
        num_elements1 = np.prod(shape1)
        num_elements2 = np.prod(shape2)

        tenval1 = np.arange(num_elements1).reshape(shape1)
        tenval2 = np.arange(num_elements2).reshape(shape2) + 6

        ten1 = self.declare_variable('ten1', val=tenval1)
        ten2 = self.declare_variable('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', csdl.cross(ten1, ten2, axis=3))


class ErrorIncorrectAxisIndex(Model):
    def define(self):
        # Creating two tensors
        shape = (2, 5, 4, 3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        ten1 = self.declare_variable('ten1', val=tenval1)
        ten2 = self.declare_variable('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', csdl.cross(ten1, ten2, axis=2))
