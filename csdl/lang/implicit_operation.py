from typing import Dict, List, Set, Union
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.output import Output
from csdl.lang.operation import Operation
from csdl.solvers.linear_solver import LinearSolver
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.nonlinear.newton import NewtonSolver
from csdl.solvers.nonlinear.broyden import BroydenSolver
import numpy as np
try:
    from csdl.lang.model import Model
except ImportError:
    pass


class ImplicitOperation(Operation):
    """
    Class for solving implicit functions using the specified solvers
    """

    def __init__(
        self,
        model: 'Model',
        out_res_map: Dict[str, Output],
        # allow Output types for exposed intermediate variables
        res_out_map: Dict[str, Union[DeclaredVariable, Output]],
        out_in_map: Dict[str, List[DeclaredVariable]],
        *args,
        expose: List[str] = [],
        defaults: Dict[str, np.ndarray] = dict(),
        nonlinear_solver: Union[NonlinearSolver, None] = None,
        linear_solver: Union[LinearSolver, None] = None,
        **kwargs,
    ):
        self.nouts = len(out_res_map.keys())
        in_vars: Set[DeclaredVariable] = set()
        # for _, v in out_in_map.items():
        #     in_vars = in_vars.union(set(v))
        self.nargs = len(in_vars)
        super().__init__(*args, **kwargs)
        self._model: Model = model
        if linear_solver is None and isinstance(
                nonlinear_solver, (NewtonSolver, BroydenSolver)):
            raise ValueError(
                "A linear solver is required when specifying a Newton or Broyden solver"
            )
        self.res_out_map: Dict[str, Union[DeclaredVariable, Output]] = res_out_map
        self.out_res_map: Dict[str, Output] = out_res_map
        self.out_in_map: Dict[str, List[DeclaredVariable]] = out_in_map
        self.expose: List[str] = expose
        self.nonlinear_solver: Union[NonlinearSolver, None] = nonlinear_solver
        self.linear_solver: Union[LinearSolver, None] = linear_solver
        self.defaults = defaults
