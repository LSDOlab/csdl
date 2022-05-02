import numpy as np
from csdl import Model, GraphRepresentation
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
        self.register_output('einsum_inner1',
                             csdl.einsum(vec, vec, subscripts='i,i->'))


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
        self.register_output('einsum_inner2',
                             csdl.einsum(
                                 tens,
                                 vec,
                                 subscripts='ijk,j->ik',
                             ))


class ExampleOuterVectorVector(Model):
    """
    :param var: a
    :param var: einsum_outer1
    """
    def define(self):
        a = np.arange(4)
        vec = self.declare_variable('a', val=a)

        # Outer Product of 2 vectors
        self.register_output('einsum_outer1',
                             csdl.einsum(vec, vec, subscripts='i,j->ij'))


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
            'einsum_outer2', csdl.einsum(
                tens,
                vec,
                subscripts='hij,k->hijk',
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

        # Transpose of a matrix
        self.register_output('einsum_reorder1',
                             csdl.einsum(mat, subscripts='ij->ji'))


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

        # Transpose of a tensor
        self.register_output('einsum_reorder2',
                             csdl.einsum(tens, subscripts='ijk->kji'))


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
                             csdl.einsum(
                                 vec,
                                 subscripts='i->',
                             ))


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
        self.register_output('einsum_summ2',
                             csdl.einsum(
                                 tens,
                                 subscripts='ijk->',
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
        self.register_output('einsum_special1',
                             csdl.einsum(vec, vec, subscripts='i,j->j'))


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
        self.register_output('einsum_special2',
                             csdl.einsum(vec, vec, subscripts='i,j->'))


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
            csdl.einsum(
                vec,
                vec,
                subscripts='i,i->',
                partial_format='sparse',
            ))


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
            csdl.einsum(tens,
                        vec,
                        subscripts='ijk,j->ik',
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
            csdl.einsum(
                vec,
                vec,
                subscripts='i,j->ij',
                partial_format='sparse',
            ))


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
            csdl.einsum(tens,
                        vec,
                        subscripts='hij,k->hijk',
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
            csdl.einsum(
                mat,
                subscripts='ij->ji',
                partial_format='sparse',
            ))


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
            csdl.einsum(
                tens,
                subscripts='ijk->kji',
                partial_format='sparse',
            ))


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
            csdl.einsum(vec, subscripts='i->', partial_format='sparse'))


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
            csdl.einsum(
                tens,
                subscripts='ijk->',
                partial_format='sparse',
            ))


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
            csdl.einsum(
                vec,
                vec,
                subscripts='i,j->j',
                partial_format='sparse',
            ))


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
            csdl.einsum(
                vec,
                vec,
                subscripts='i,j->',
                partial_format='sparse',
            ))
