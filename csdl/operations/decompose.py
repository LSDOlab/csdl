from csdl.core.standard_operation import StandardOperation
from csdl.core.node import Node
from csdl.core.output import Output
import numpy as np


class decompose(StandardOperation):

    def __init__(self, *args, **kwargs):
        self.nargs = 1
        self.nouts = 1
        super().__init__(*args, **kwargs)
        self.outs = []
        self._key_out_pairs: Dict[Tuple[Tuple[int]], Output] = dict()
        self.src_indices: Dict[Output, np.ndarray] = dict()
