from csdl.lang.standard_operation import StandardOperation
from csdl.lang.node import Node
from csdl.lang.output import Output
import numpy as np
from typing import Dict, Tuple


class decompose(StandardOperation):

    def __init__(self, *args, **kwargs):
        self.nargs = 1
        self.nouts = 1
        super().__init__(*args, **kwargs)
        self.outs = ()
        self._key_out_pairs: Dict[Tuple[Tuple[int]], Output] = dict()
        self.src_indices: Dict[Output, np.ndarray] = dict()
        self.properties['linear'] = True
