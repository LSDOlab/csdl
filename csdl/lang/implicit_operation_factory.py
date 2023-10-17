from typing import Tuple, List, Dict, Union
from csdl.lang.output import Output

# from csdl.lang.intermediate_representation import IntermediateRepresentation
from csdl.lang.variable import Variable
from csdl.rep.graph_representation import GraphRepresentation
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.linear_solver import LinearSolver
from csdl.utils.check_default_val_type import check_default_val_type
import numpy as np
try:
    from csdl.lang.model import Model
except ImportError:
    pass
from collections import OrderedDict


class ImplicitOperationFactory(object):

    def __init__(self, parent: 'Model', model: 'Model', use_vjps: bool):
        self.parent: 'Model' = parent
        self.model: 'Model' = model
        self.residuals: List[str] = []
        self.nonlinear_solver: Union[NonlinearSolver, None] = None
        self.linear_solver: Union[LinearSolver, None] = None
        self.brackets: Dict[str, Tuple[Union[int, float, np.ndarray,
                                             Variable],
                                       Union[int, float, np.ndarray,
                                             Variable]]] = dict()
        self.states: OrderedDict[str, dict] = OrderedDict()
        self.use_vjps: bool = use_vjps

    def declare_state(
        self,
        state: str,
        bracket: Union[Tuple[Union[int, float, np.ndarray, Variable],
                             Union[int, float, np.ndarray, Variable]],
                       None] = None,
        val=1.0,
        units=None,
        desc='',
        *,
        residual: str,
    ):
        self.residuals.append(residual)
        if bracket is not None:
            self.brackets[state] = bracket
        self.states[state] = dict(
            val=check_default_val_type(val),
            units=units,
            desc=desc,
        )

    # TODO: what is defaults?
    # TODO: where to define maxiter?
    # TODO: where to define tol?
    def __call__(
            self,
            *arguments: Variable,
            expose: List[str] = [],
            defaults: Dict[str, Union[int, float, np.ndarray]] = dict(),
    ):
        return self.apply(
            *arguments,
            expose=expose,
            defaults=defaults,
        )

    def apply(
        self,
        *arguments: Variable,
        expose: List[str] = [],
        defaults: Dict[str, Union[int, float, np.ndarray]] = dict(),
    ) -> Union[Output, Tuple[Output, ...]]:
        if len(self.brackets) > 0:
            return self.parent._bracketed_search(
                self.states,
                self.residuals,
                self.model,
                self.brackets,
                self.use_vjps,
                *arguments,
                expose=expose,
            )
        else:
            return self.parent._implicit_operation(
                self.states,
                self.use_vjps,
                *arguments,
                residuals=self.residuals,
                model=self.model,
                nonlinear_solver=self.nonlinear_solver,
                linear_solver=self.linear_solver,
                expose=expose,
                defaults=defaults,
            )
