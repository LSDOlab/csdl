from typing import Dict, List, Set
from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.core.operation import Operation
from csdl.solvers.linear_solver import LinearSolver
from csdl.solvers.nonlinear_solver import NonlinearSolver


class ImplicitOperation(Operation):
    """
    Class for solving implicit functions using the specified solvers
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
