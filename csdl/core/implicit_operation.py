from typing import Dict, List, Set
from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.core.operation import Operation
from csdl.solvers.linear_solver import LinearSolver
from csdl.solvers.nonlinear_solver import NonlinearSolver


class ImplicitOperation(Operation):
    """
    Class for creating `ImplicitOperation` objects that solve implicit
    functions.
    """
    def __init__(
        self,
        model,
        nonlinear_solver: NonlinearSolver,
        linear_solver: LinearSolver,
        out_res_map: Dict[str, Output],
        res_out_map: Dict[str, Variable],
        out_in_map: Dict[str, List[Variable]],
        *args,
        **kwargs,
    ):
        self.nouts = len(out_res_map.keys())
        in_vars: Set[Variable] = set()
        for _, v in out_in_map.items():
            in_vars = in_vars.union(set(v))
        self.nargs = len(in_vars)
        super().__init__(*args, **kwargs)
        self._model = model
        self.nonlinear_solver = nonlinear_solver
        self.linear_solver = linear_solver
        self.res_out_map: Dict[str, Variable] = res_out_map
        self.out_res_map: Dict[str, Output] = out_res_map
        self.out_in_map: Dict[str, List[Variable]] = out_in_map


from typing import Tuple, Union
import numpy as np


class BracketedSearchOperation(Operation):
    """
    Class for creating `ImplicitOperation` objects that solve implicit
    functions.
    """
    def __init__(
        self,
        model,
        out_res_map: Dict[str, Output],
        res_out_map: Dict[str, Variable],
        out_in_map: Dict[str, List[Variable]],
        brackets: Dict[str, Tuple[Union[float, np.ndarray]]],
        *args,
        **kwargs,
    ):
        self.nouts = len(out_res_map.keys())
        in_vars: Set[Variable] = set()
        for _, v in out_in_map.items():
            in_vars = in_vars.union(set(v))
        self.nargs = len(in_vars)
        super().__init__(*args, **kwargs)
        self._defined = False
        self._model = model
        self.res_out_map: Dict[str, Variable] = res_out_map
        self.out_res_map: Dict[str, Output] = out_res_map
        self.out_in_map: Dict[str, List[Variable]] = out_in_map
        for k, v in brackets.items():
            if len(v) != 2:
                raise ValueError(
                    "Bracket for state {} is not a tuple of two values".
                    format(k))
            if isinstance(v[0], np.ndarray) and isinstance(
                    v[1], np.ndarray):
                if v[0].shape != v[1].shape:
                    raise ValueError(
                        "Bracket values for {} are not the same shape; {} != {}"
                        .format(k, v[0].shape, v[1].shape))
            elif not isinstance(v[0], float) and not isinstance(
                    v[1], float):
                raise TypeError(
                    "Bracket values for {} are not of the same type; type must be float or ndarray"
                    .format(k))

        self.brackets: Dict[str, Tuple[Union[float,
                                             np.ndarray]]] = brackets
