from typing import Tuple, List, Dict, Union, Optional
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

    def __init__(self, parent: 'Model', model: 'Model', name: Optional[str]):
        self.name: Optional[str] = name
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

    def declare_state(
        self,
        state: str,
        bracket: Union[Tuple[Union[int, float, np.ndarray, Variable],
                             Union[int, float, np.ndarray, Variable]],
                       None] = None,
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
        self.residuals.append(residual)
        if bracket is not None:
            self.brackets[state] = bracket
        self.states[state] = dict(
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
        hybrid: bool = False,
        backend: str = '',
    ) -> Union[Output, Tuple[Output, ...]]:
        if len(self.brackets) > 0:
            if hybrid == True:
                return self.parent._hybrid_bracketed_search(
                    self.states,
                    self.name,
                    self.residuals,
                    self.model,
                    self.brackets,
                    *arguments,
                    expose=expose,
                    backend=backend,
                )
            else:
                return self.parent._bracketed_search(
                    self.states,
                    self.residuals,
                    self.model,
                    self.brackets,
                    *arguments,
                    expose=expose,
                )
        else:
            if hybrid == True:
                if self.name is None:
                    raise ValueError('Name is required for hybrid implicit operations')
                if not isinstance(backend, str) or len(backend) == 0:
                    raise ValueError(
                        'Argument `backend` must be a string specifying a Python package that provides a compiler back end.'
                    )
                return self.parent._hybrid_implicit_operation(
                    self.states,
                    self.name,
                    *arguments,
                    residuals=self.residuals,
                    model=self.model,
                    nonlinear_solver=self.nonlinear_solver,
                    linear_solver=self.linear_solver,
                    expose=expose,
                    defaults=defaults,
                    backend=backend,
                )
            else:
                return self.parent._implicit_operation(
                    self.states,
                    *arguments,
                    residuals=self.residuals,
                    model=self.model,
                    nonlinear_solver=self.nonlinear_solver,
                    linear_solver=self.linear_solver,
                    expose=expose,
                    defaults=defaults,
                )
