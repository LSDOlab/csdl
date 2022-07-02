from typing import Dict, List, Set, Tuple, Union
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.output import Output
from csdl.lang.implicit_operation import ImplicitOperation

import numpy as np


class BracketedSearchOperation(ImplicitOperation):
    """
    Class for solving implicit functions using a bracketed search
    """

    def __init__(
        self,
        model,
        out_res_map: Dict[str, Output],
        # allow Output types for exposed intermediate variables
        res_out_map: Dict[str, Union[DeclaredVariable, Output]],
        out_in_map: Dict[str, List[DeclaredVariable]],
        *args,
        expose: List[str] = [],
        brackets: Dict[str, Tuple[np.ndarray, np.ndarray]] = dict(),
        maxiter: int = 1000,
        tol: float = 1e-7,
        **kwargs,
    ):
        super().__init__(
            model,
            out_res_map,
            res_out_map,
            out_in_map,
            *args,
            expose=expose,
            **kwargs,
        )
        in_vars: Set[DeclaredVariable] = set()
        for _, v in out_in_map.items():
            in_vars = in_vars.union(set(v))
        self._model = model
        self.brackets: Dict[str, Tuple[Union[np.ndarray, Variable],
                                       Union[np.ndarray,
                                             Variable]]] = brackets
        self.maxiter: int = maxiter
        self.tol: float = tol
