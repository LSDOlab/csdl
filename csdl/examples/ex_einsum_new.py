import numpy as np
from csdl import Model
import csdl


# Note: Expansion is not possible with einsum
class ExampleInnerVectorVector(Model):
    """
    :param var: a
    :param var: einsum_inner1
    """
    def define(self):
        a = np.arange(4)

        vec = self.declare_variable('a', val=a)

        # Inner Product of 2 vectors
        self.register_output(
            'einsum_inner1',
            csdl.einsum_new_api(vec, vec, operation=[(0, ), (0, )]))


class ExampleInnerTensorVector(Model):
    """
    :param var: a
    :param var: c
    :param var: einsum_inner2
    """
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        # Inner Product of a tensor and a vector
        self.register_output(
            'einsum_inner2',
            csdl.einsum_new_api(
                tens,
                vec,
                operation=[('rows', 0, 1), (0, ), ('rows', 1)],
            ))


class ExampleOuterVectorVector(Model):
    """
    :param var: a
    :param var: einsum_outer1
    """
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        self.register_output(
            'einsum_outer1',
            csdl.einsum_new_api(
                vec,
                vec,
                operation=[('rows', ), ('cols', ), ('rows', 'cols')],
            ))


class ExampleOuterTensorVector(Model):
    """
    :param var: a
    :param var: c
    :param var: einsum_outer2
    """
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        # Outer Product of a tensor and a vector
        self.register_output(
            'einsum_outer2',
            csdl.einsum_new_api(
                tens,
                vec,
                operation=[(0, 1, 30), (2, ), (0, 1, 30, 2)],
            ))


class ExampleReorderMatrix(Model):
    """
    :param var: b
    :param var: einsum_reorder1
    """
    def define(self):
        shape2 = (5, 4)
        b = np.arange(20).reshape(shape2)
        mat = self.declare_variable('b', val=b)

        # reorder of a matrix
        self.register_output(
            'einsum_reorder1',
            csdl.einsum_new_api(mat, operation=[(46, 99), (99, 46)]))


class ExampleReorderTensor(Model):
    """
    :param var: c
    :param var: einsum_reorder2
    """
    def define(self):

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        # reorder of a tensor
        self.register_output(
            'einsum_reorder2',
            csdl.einsum_new_api(
                tens,
                operation=[(33, 66, 99), (99, 66, 33)],
            ))


class ExampleVectorSummation(Model):
    """
    :param var: a
    :param var: einsum_summ1
    """
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Summation of all the entries of a vector
        self.register_output('einsum_summ1',
                             csdl.einsum_new_api(vec, operation=[(33, )]))


class ExampleTensorSummation(Model):
    """
    :param var: c
    :param var: einsum_summ2
    """
    def define(self):
        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        # Summation of all the entries of a tensor
        self.register_output(
            'einsum_summ2',
            csdl.einsum_new_api(
                tens,
                operation=[(33, 66, 99)],
            ))


class ExampleMultiplicationSum(Model):
    """
    :param var: a
    :param var: einsum_special1
    """
    def define(self):

        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Special operation: summation of all the entries of first
        # vector and scalar multiply the second vector with the computed
        # sum
        self.register_output(
            'einsum_special1',
            csdl.einsum_new_api(
                vec,
                vec,
                operation=[(1, ), (2, ), (2, )],
            ))


class ExampleMultipleVectorSum(Model):
    """
    :param var: a
    :param var: einsum_special2
    """
    def define(self):

        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Special operation: sum all the entries of the first and second
        # vector to a single scalar
        self.register_output(
            'einsum_special2',
            csdl.einsum_new_api(vec, vec, operation=[(1, ), (2, )]))


# All the above operations done with sparse partials (memory
# efficient when the partials are sparse and large)


class ExampleInnerVectorVectorSparse(Model):
    """
    :param var: a
    :param var: einsum_inner1_sparse_derivs
    """
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        self.register_output(
            'einsum_inner1_sparse_derivs',
            csdl.einsum_new_api(vec,
                                vec,
                                operation=[(0, ), (0, )],
                                partial_format='sparse'))


class ExampleInnerTensorVectorSparse(Model):
    """
    :param var: a
    :param var: c
    :param var: einsum_inner2_sparse_derivs
    """
    def define(self):

        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        self.register_output(
            'einsum_inner2_sparse_derivs',
            csdl.einsum_new_api(tens,
                                vec,
                                operation=[
                                    ('rows', 0, 1),
                                    (0, ),
                                    ('rows', 1),
                                ],
                                partial_format='sparse'))


class ExampleOuterVectorVectorSparse(Model):
    """
    :param var: a
    :param var: einsum_outer1_sparse_derivs
    """
    def define(self):

        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        self.register_output(
            'einsum_outer1_sparse_derivs',
            csdl.einsum_new_api(vec,
                                vec,
                                operation=[('rows', ), ('cols', ),
                                           ('rows', 'cols')],
                                partial_format='sparse'))


class ExampleOuterTensorVectorSparse(Model):
    """
    :param var: a
    :param var: c
    :param var: einsum_outer2_sparse_derivs
    """
    def define(self):

        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        self.register_output(
            'einsum_outer2_sparse_derivs',
            csdl.einsum_new_api(tens,
                                vec,
                                operation=[
                                    (0, 1, 30),
                                    (2, ),
                                    (0, 1, 30, 2),
                                ],
                                partial_format='sparse'))


class ExampleReorderMatrixSparse(Model):
    """
    :param var: b
    :param var: einsum_reorder1_sparse_derivs
    """
    def define(self):

        shape2 = (5, 4)
        b = np.arange(20).reshape(shape2)
        mat = self.declare_variable('b', val=b)

        self.register_output(
            'einsum_reorder1_sparse_derivs',
            csdl.einsum_new_api(mat,
                                operation=[(46, 99), (99, 46)],
                                partial_format='sparse'))


class ExampleReorderTensorSparse(Model):
    """
    :param var: c
    :param var: einsum_reorder2_sparse_derivs
    """
    def define(self):

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        self.register_output(
            'einsum_reorder2_sparse_derivs',
            csdl.einsum_new_api(tens,
                                operation=[(33, 66, 99), (99, 66, 33)],
                                partial_format='sparse'))


class ExampleVectorSummationSparse(Model):
    """
    :param var: a
    :param var: einsum_summ1_sparse_derivs
    """
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        self.register_output(
            'einsum_summ1_sparse_derivs',
            csdl.einsum_new_api(vec,
                                operation=[(33, )],
                                partial_format='sparse'))


class ExampleTensorSummationSparse(Model):
    """
    :param var: c
    :param var: einsum_summ2_sparse_derivs
    """
    def define(self):
        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_variable('c', val=c)

        self.register_output(
            'einsum_summ2_sparse_derivs',
            csdl.einsum_new_api(tens,
                                operation=[(33, 66, 99)],
                                partial_format='sparse'))


class ExampleMultiplicationSumSparse(Model):
    """
    :param var: a
    :param var: einsum_special1_sparse_derivs
    """
    def define(self):

        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        self.register_output(
            'einsum_special1_sparse_derivs',
            csdl.einsum_new_api(vec,
                                vec,
                                operation=[(1, ), (2, ), (2, )],
                                partial_format='sparse'))


class ExampleMultipleVectorSumSparse(Model):
    """
    :param var: a
    :param var: einsum_special2_sparse_derivs
    """
    def define(self):

        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        self.register_output(
            'einsum_special2_sparse_derivs',
            csdl.einsum_new_api(vec,
                                vec,
                                operation=[
                                    (1, ),
                                    (2, ),
                                ],
                                partial_format='sparse'))
