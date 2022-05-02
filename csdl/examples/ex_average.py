from csdl import Model, GraphRepresentation
import csdl
import numpy as np


class ExampleSingleVector(Model):
    """
    :param var: v1
    :param var: single_vector_average
    """
    def define(self):
        n = 3

        # Declare a vector of length 3 as input
        v1 = self.declare_variable('v1', val=np.arange(n))

        # Output the average of all the elements of the vector v1
        self.register_output('single_vector_average', csdl.average(v1))


class ExampleSingleMatrix(Model):
    """
    :param var: M1
    :param var: single_matrix_average
    """
    def define(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_variable(
            'M1',
            val=np.arange(n * m).reshape((n, m)),
        )

        # Output the average of all the elements of the matrix M1
        self.register_output(
            'single_matrix_average',
            csdl.average(M1),
        )


class ExampleSingleTensor(Model):
    """
    :param var: T1
    :param var: single_tensor_average
    """
    def define(self):
        n = 3
        m = 6
        p = 7
        q = 10

        # Declare a tensor of shape 3x6x7x10 as input
        T1 = self.declare_variable(
            'T1',
            val=np.arange(n * m * p * q).reshape((n, m, p, q)),
        )

        # Output the average of all the elements of the tensor T1
        self.register_output(
            'single_tensor_average',
            csdl.average(T1),
        )


class ExampleMultipleVector(Model):
    """
    :param var: v1
    :param var: v2
    :param var: multiple_vector_average
    """
    def define(self):
        n = 3

        # Declare a vector of length 3 as input
        v1 = self.declare_variable('v1', val=np.arange(n))

        # Declare another vector of length 3 as input
        v2 = self.declare_variable('v2', val=np.arange(n, 2 * n))

        # Output the elementwise average of vectors v1 and v2
        self.register_output(
            'multiple_vector_average',
            csdl.average(v1, v2),
        )


class ExampleMultipleMatrix(Model):
    """
    :param var: M1
    :param var: M2
    :param var: multiple_matrix_average
    """
    def define(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_variable(
            'M1',
            val=np.arange(n * m).reshape((n, m)),
        )

        # Declare another matrix of shape 3x6 as input
        M2 = self.declare_variable(
            'M2',
            val=np.arange(n * m, 2 * n * m).reshape((n, m)),
        )

        # Output the elementwise average of matrices M1 and M2
        self.register_output(
            'multiple_matrix_average',
            csdl.average(M1, M2),
        )


class ExampleMultipleTensor(Model):
    """
    :param var: T1
    :param var: T2
    :param var: multiple_tensor_average
    """
    def define(self):
        n = 3
        m = 6
        p = 7
        q = 10

        # Declare a tensor of shape 3x6x7x10 as input
        T1 = self.declare_variable(
            'T1',
            val=np.arange(n * m * p * q).reshape((n, m, p, q)),
        )

        # Declare another tensor of shape 3x6x7x10 as input
        T2 = self.declare_variable(
            'T2',
            val=np.arange(n * m * p * q, 2 * n * m * p * q).reshape(
                (n, m, p, q)),
        )
        # Output the elementwise average of tensors T1 and T2
        self.register_output('multiple_tensor_average',
                             csdl.average(T1, T2))


class ExampleSingleMatrixAlong0(Model):
    """
    :param var: M1
    :param var: single_matrix_average_along_0
    """
    def define(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_variable(
            'M1',
            val=np.arange(n * m).reshape((n, m)),
        )

        # Output the axiswise average of matrix M1 along the columns
        self.register_output(
            'single_matrix_average_along_0',
            csdl.average(M1, axes=(0, )),
        )


class ExampleSingleMatrixAlong1(Model):
    """
    :param var: M1
    :param var: single_matrix_average_along_1
    """
    def define(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_variable(
            'M1',
            val=np.arange(n * m).reshape((n, m)),
        )

        # Output the axiswise average of matrix M1 along the columns
        self.register_output(
            'single_matrix_average_along_1',
            csdl.average(M1, axes=(1, )),
        )


class ExampleMultipleMatrixAlong0(Model):
    """
    :param var: M1
    :param var: M2
    :param var: multiple_matrix_average_along_0
    """
    def define(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_variable(
            'M1',
            val=np.arange(n * m).reshape((n, m)),
        )

        # Declare another matrix of shape 3x6 as input
        M2 = self.declare_variable(
            'M2',
            val=np.arange(n * m, 2 * n * m).reshape((n, m)),
        )

        # Output the elementwise average of the axiswise average of
        # matrices M1 ad M2 along the columns
        self.register_output(
            'multiple_matrix_average_along_0',
            csdl.average(M1, M2, axes=(0, )),
        )


class ExampleMultipleMatrixAlong1(Model):
    """
    :param var: M1
    :param var: M2
    :param var: multiple_matrix_average_along_1
    """
    def define(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_variable('M1',
                                   val=np.arange(n * m).reshape((n, m)))

        # Declare another matrix of shape 3x6 as input
        M2 = self.declare_variable('M2',
                                   val=np.arange(n * m,
                                                 2 * n * m).reshape(
                                                     (n, m)))

        # Output the elementwise average of the axiswise average of
        # matrices M1 ad M2 along the columns
        self.register_output(
            'multiple_matrix_average_along_1',
            csdl.average(M1, M2, axes=(1, )),
        )
