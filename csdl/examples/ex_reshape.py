from csdl import Model, GraphRepresentation
import csdl
import numpy as np


class ExampleTensor2Vector(Model):
    """
    :param var: in1
    :param var: reshape_tensor2vector
    """
    def define(self):
        shape = (2, 3, 4, 5)
        size = 2 * 3 * 4 * 5

        # Both vector or tensors need to be numpy arrays
        tensor = np.arange(size).reshape(shape)
        vector = np.arange(size)

        # in1 is a tensor having shape = (2, 3, 4, 5)
        in1 = self.declare_variable('in1', val=tensor)

        # in1 is being reshaped from shape = (2, 3, 4, 5) to a vector
        # having size = 2 * 3 * 4 * 5
        self.register_output('reshape_tensor2vector',
                             csdl.reshape(in1, new_shape=(size, )))


class ExampleVector2Tensor(Model):
    """
    :param var: in2
    :param var: reshape_vector2tensor
    """
    def define(self):
        shape = (2, 3, 4, 5)
        size = 2 * 3 * 4 * 5

        # Both vector or tensors need to be numpy arrays
        tensor = np.arange(size).reshape(shape)
        vector = np.arange(size)

        # in2 is a vector having a size of 2 * 3 * 4 * 5
        in2 = self.declare_variable('in2', val=vector)

        # in2 is being reshaped from size =  2 * 3 * 4 * 5 to a tenßsor
        # having shape = (2, 3, 4, 5)
        self.register_output('reshape_vector2tensor',
                             csdl.reshape(in2, new_shape=shape))
