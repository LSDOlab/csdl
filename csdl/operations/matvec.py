from csdl.lang.variable import Variable
from csdl.lang.standard_operation import StandardOperation
from csdl.lang.variable import Variable
from scipy.sparse import spmatrix


class matvec(StandardOperation):

    def __init__(self, *args, mat, **kwargs):
        self.nouts = 1
        self.nargs = 2
        super().__init__(*args, **kwargs)
        self.literals['sparsemtx'] = None
        if isinstance(mat, Variable):
            self.dependencies = [mat, self.dependencies[0]]
        else:
            self.literals['sparsemtx'] = mat
