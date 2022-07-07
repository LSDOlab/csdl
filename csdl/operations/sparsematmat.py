from csdl.lang.standard_operation import StandardOperation


class sparsematmat(StandardOperation):
    def __init__(self, *args, sparse_mat, **kwargs):
        self.nouts = 1
        self.nargs = 1
        super().__init__(*args, **kwargs)
        self.properties['iterative'] = False
        self.literals['sparse_mat'] = sparse_mat
