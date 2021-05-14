from csdl import Model
import csdl
import numpy as np


class ExampleMatVecProduct(Model):
    """
    :param var: mat1
    :param var: vec1
    :param var: MatVec
    """
    def define(self):
        m = 3
        n = 4

        # Shape of the first matrix (3,2)
        shape1 = (m, n)

        # Shape of the second matrix (2,4)
        shape2 = (n, )

        # Creating the values of both matrices
        val1 = np.arange(m * n).reshape(shape1)
        val2 = np.arange(n).reshape(shape2)

        # Declaring the input matrix and input vector
        mat1 = self.declare_variable('mat1', val=val1)
        vec1 = self.declare_variable('vec1', val=val2)

        # Creating the output for matrix-vector multiplication
        self.register_output('MatVec', csdl.matvec(mat1, vec1))


class ErrorMatrixVectorIncompatibleShapes(Model):
    def define(self):
        m = 3
        n = 4

        # Shape of the first matrix (3,2)
        shape1 = (m, n)

        # Shape of the second matrix (2,4)
        shape2 = (m, )

        # Creating the values of both matrices
        val1 = np.arange(m * n).reshape(shape1)
        val2 = np.arange(n).reshape(shape2)

        # Declaring the input matrix and input vector
        mat1 = self.declare_variable('mat1', val=val1)
        vec1 = self.declare_variable('vec1', val=val2)

        # Creating the output for matrix-vector multiplication
        self.register_output('MatVec', csdl.matvec(mat1, vec1))
