from typing import Dict, List, Set, Tuple, Union
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.output import Output
from csdl.lang.hybrid_implicit_operation import HybridImplicitOperation
from csdl.lang.variable import Variable
from csdl.lang.model import Model
from csdl.solvers.linear_solver import LinearSolver

import numpy as np

try:
    from csdl.rep.graph_representation import GraphRepresentation
except ImportError:
    pass


class M(Model):

    def initialize(self):
        self.parameters.declare('model', types=(Model))
        self.parameters.declare('input_data', types=(dict))
        self.parameters.declare('out_res_map', types=(dict))
        self.parameters.declare('linear_solver', types=(LinearSolver))
        self.parameters.declare('brackets', types=(dict))

    def define(self):
        model = self.parameters['model']
        input_data = self.parameters['input_data']
        out_res_map = self.parameters['out_res_map']
        linear_solver = self.parameters['linear_solver']
        brackets = self.parameters['brackets']
        op = self.create_implicit_operation(model)
        for output, residual in out_res_map.items():
            op.declare_state(output, residual=residual, bracket=brackets[output])
        op.linear_solver = linear_solver
        inputs = [
            self.declare_variable(input_name, shape=input_shape)
            for input_name, input_shape in input_data.items()
        ]
        outputs = op.apply(*inputs)


class HybridBracketedSearchOperation(HybridImplicitOperation):
    """
    Class for solving implicit functions using a bracketed search
    """

    def __init__(
        self,
        mode: str,
        name: str,
        model: 'Model',
        rep: 'GraphRepresentation',
        out_res_map: Dict[str, Output],
        # allow Output types for exposed intermediate variables
        res_out_map: Dict[str, DeclaredVariable],
        out_in_map: Dict[str, List[DeclaredVariable]],
        exp_in_map: Dict[str, List[DeclaredVariable]],
        exposed_variables: Dict[str, Output],
        exposed_residuals: Set[str],
        *args,
        expose: List[str] = [],
        brackets: Dict[str, Tuple[Union[np.ndarray, Variable],
                                      Union[np.ndarray,
                                            Variable]]] = dict(),
        maxiter: int = 1000,
        tol: float = 1e-7,
        **kwargs,
    ):
        super().__init__(
            mode,
            name,
            model,
            rep,
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_variables,
            exposed_residuals,
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
        for l, u in self.brackets.values():
            if isinstance(l, Variable):
                self.add_dependency_node(l)
            if isinstance(u, Variable):
                self.add_dependency_node(u)

    def _build_internal_simulator(self):
        exec(f'from {self.backend} import Simulator')
        if self.mode == 'explicit':
            self.sim = Simulator(self.rep)
        elif self.mode == 'implicit':
            # TODO: apply middle end optimizations to `rep`
            rep = GraphRepresentation(
                M(
                    model=self.model,
                    input_data=self.input_data,
                    out_res_map=self.out_res_map,
                    linear_solver=self.linear_solver,
                    brackets=self.brackets,
                ), )
            self.sim = Simulator(rep)
