from csdl import Model, GraphRepresentation
import csdl
from scipy.sparse import coo_matrix
import numpy as np


class ExampleMatMat(Model):

    def define(self):
        mat = np.ones([4, 4])
        mat1 = self.create_input(name='matrix', val=mat)

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data2 = np.array([1., 2., 0., 12.])
        mat2 = coo_matrix((data2, (row, col)), shape=(4, 4)).tocsc()

        actuating_points = csdl.sparsematmat(mat1, sparse_mat=mat2)

        out = self.register_output('out', var=actuating_points)
