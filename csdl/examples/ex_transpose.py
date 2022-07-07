from csdl import Model, GraphRepresentation
import csdl
import numpy as np


class ExampleMatrix(Model):
    """
    :param var: Mat
    :param var: matrix_transpose
    """
    def define(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_variable(
            'Mat',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Compute the transpose of mat
        self.register_output('matrix_transpose', csdl.transpose(mat))


class ExampleTensor(Model):
    """
    :param var: Tens
    :param var: tensor_transpose
    """
    def define(self):

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_variable(
            'Tens',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute the transpose of tens
        self.register_output('tensor_transpose', csdl.transpose(tens))
