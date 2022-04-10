from typing import Tuple, List, Dict

# from csdl.core.intermediate_representation import IntermediateRepresentation
from csdl.core.variable import Variable
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.linear_solver import LinearSolver
from csdl.utils.check_default_val_type import check_default_val_type
import numpy as np


class ImplicitOperationFactory(object):

    def __init__(self, parent: 'Model', model: 'Model'):
        from csdl.core.model import Model
        self.parent: 'Model' = parent
        self.model: 'Model' = model
        self.states: List[str] = []
        self.residuals: List[str] = []
        self.nonlinear_solver: NonlinearSolver | None = None
        self.linear_solver: LinearSolver | None = None
        self.brackets: Dict[str,
                            Tuple[int | float | np.ndarray,
                                  int | float | np.ndarray]] = dict()
        self.implicit_metadata: Dict[str, dict] = dict()

    def declare_state(
        self,
        state: str,
        bracket: Tuple[int | float | np.ndarray,
                       int | float | np.ndarray] = None,
        val=1.0,
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
        res_units=None,
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=1.0,
        *,
        residual: str,
    ):
        self.states.append(state)
        self.residuals.append(residual)
        if bracket is not None:
            self.brackets[state] = bracket
        self.implicit_metadata[state] = dict(
            val=check_default_val_type(val),
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            distributed=distributed,
            res_units=res_units,
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            res_ref=res_ref,
        )

    def __call__(
            self,
            *arguments: Variable,
            expose: List[str] = [],
            defaults: Dict[str, int | float | np.ndarray] = dict(),
    ):
        if len(self.brackets) > 0:
            return self.parent._bracketed_search(
                self.implicit_metadata,
                *arguments,
                states=self.states,
                residuals=self.residuals,
                model=self.model,
                brackets=self.brackets,
                expose=expose,
            )
        else:
            return self.parent._implicit_operation(
                self.implicit_metadata,
                *arguments,
                states=self.states,
                residuals=self.residuals,
                model=self.model,
                nonlinear_solver=self.nonlinear_solver,
                linear_solver=self.linear_solver,
                expose=expose,
                defaults=defaults,
            )
