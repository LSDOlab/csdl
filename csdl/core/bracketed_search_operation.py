from typing import Dict, List, Set, Tuple
from csdl.core.declared_variable import DeclaredVariable
from csdl.core.output import Output
from csdl.core.operation import Operation

import numpy as np


class BracketedSearchOperation(Operation):
    """
    Class for solving implicit functions using a bracketed search
    """

    def __init__(
        self,
        model,
        out_res_map: Dict[str, Output],
        res_out_map: Dict[str, DeclaredVariable],
        out_in_map: Dict[str, List[DeclaredVariable]],
        brackets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        expose: List[str] = [],
        maxiter: int = 100,
        *args,
        **kwargs,
    ):
        self.nouts = len(out_res_map.keys())
        in_vars: Set[DeclaredVariable] = set()
        for _, v in out_in_map.items():
            in_vars = in_vars.union(set(v))
        self.nargs = len(in_vars)
        super().__init__(*args, **kwargs)
        self._model = model
        self.res_out_map: Dict[str, DeclaredVariable] = res_out_map
        self.out_res_map: Dict[str, Output] = out_res_map
        self.out_in_map: Dict[str, List[DeclaredVariable]] = out_in_map
        self.brackets: Dict[str, Tuple[np.ndarray,
                                       np.ndarray]] = brackets
        self.expose: List[str] = expose
        self.maxiter: int = maxiter
