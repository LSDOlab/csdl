from csdl import Model
import csdl
import numpy as np


class ExampleVectorVector(Model):
    """
    :param var: vec1
    :param var: vec2
    :param var: VecVecOuter
    """
    def define(self):

        m = 3

        # Shape of the vectors
        vec_shape = (m, )

        # Values for the two vectors
        vec1 = np.arange(m)
        vec2 = np.arange(m, 2 * m)

        # Adding the vectors to csdl
        vec1 = self.declare_variable('vec1', val=vec1)
        vec2 = self.declare_variable('vec2', val=vec2)

        # Vector-Vector Outer Product
        self.register_output('VecVecOuter', csdl.outer(vec1, vec2))


class ExampleTensorVector(Model):
    """
    :param var: vec1
    :param var: ten1
    :param var: TenVecOuter
    """
    def define(self):

        m = 3
        n = 4
        p = 5

        # Shape of the vectors
        vec_shape = (m, )

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Values for the two vectors
        vec1 = np.arange(m)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)

        # Adding the vector and tensor to csdl
        vec1 = self.declare_variable('vec1', val=vec1)

        ten1 = self.declare_variable('ten1', val=ten1)

        # Tensor-Vector Outer Product specifying the first axis for Vector and Tensor
        self.register_output('TenVecOuter', csdl.outer(ten1, vec1))


class ExampleTensorTensor(Model):
    """
    :param var: ten1
    :param var: ten2
    :param var: TenTenOuter
    """
    def define(self):

        m = 3
        n = 4
        p = 5

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the tensors to csdl
        ten1 = self.declare_variable('ten1', val=ten1)
        ten2 = self.declare_variable('ten2', val=ten2)

        # Tensor-Tensor Outer Product specifying the first and last axes
        self.register_output('TenTenOuter', csdl.outer(ten1, ten2))
