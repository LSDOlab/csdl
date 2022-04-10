from typing import Dict, List, Set
from csdl.core.declared_variable import DeclaredVariable
from csdl.core.output import Output
from csdl.core.operation import Operation
from csdl.solvers.linear_solver import LinearSolver
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.nonlinear.newton import NewtonSolver
from csdl.solvers.nonlinear.broyden import BroydenSolver
import numpy as np


class ImplicitOperation(Operation):
    """
    Class for solving implicit functions using the specified solvers
    """

    def __init__(
            self,
            model,
            nonlinear_solver: NonlinearSolver,
            linear_solver: LinearSolver | None,
            out_res_map: Dict[str, Output],
            res_out_map: Dict[str, DeclaredVariable],
            out_in_map: Dict[str, List[DeclaredVariable]],
            expose: List[str] = [],
            defaults: Dict[str, np.ndarray] = dict(),
            *args,
            **kwargs,
    ):
        self.nouts = len(out_res_map.keys())
        in_vars: Set[DeclaredVariable] = set()
        for _, v in out_in_map.items():
            in_vars = in_vars.union(set(v))
        self.nargs = len(in_vars)
        super().__init__(*list(in_vars), **kwargs)
        from csdl.core.model import Model
        self._model: Model = model
        if linear_solver is None and isinstance(
                nonlinear_solver, (NewtonSolver, BroydenSolver)):
            raise ValueError(
                "A linear solver is required when specifying a Newton or Broyden solver"
            )
        self.nonlinear_solver = nonlinear_solver
        self.linear_solver = linear_solver
        self.res_out_map: Dict[str, DeclaredVariable] = res_out_map
        self.out_res_map: Dict[str, Output] = out_res_map
        self.out_in_map: Dict[str, List[DeclaredVariable]] = out_in_map
        self.expose: List[str] = expose
        self.defaults = defaults
