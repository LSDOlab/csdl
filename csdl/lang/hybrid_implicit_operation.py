from csdl.lang.model import Model
from csdl.rep.graph_representation import GraphRepresentation
from csdl.lang.custom_explicit_operation import CustomExplicitOperation
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.linear_solver import LinearSolver
from csdl.solvers.nonlinear.newton import NewtonSolver
from csdl.solvers.nonlinear.broyden import BroydenSolver

from typing import Dict, List, Set, Union, Tuple
from csdl.lang.output import Output
from csdl.lang.declared_variable import DeclaredVariable
from copy import deepcopy


class M(Model):

    def initialize(self):
        self.parameters.declare('model', types=(Model))
        self.parameters.declare('input_data', types=(dict))
        self.parameters.declare('out_res_map', types=(dict))
        self.parameters.declare('nonlinear_solver',
                                types=(NonlinearSolver))
        self.parameters.declare('linear_solver', types=(LinearSolver))

    def define(self):
        model = self.parameters['model']
        input_data = self.parameters['input_data']
        out_res_map = self.parameters['out_res_map']
        nonlinear_solver = self.parameters['nonlinear_solver']
        linear_solver = self.parameters['linear_solver']
        op = self.create_implicit_operation(model)
        for output, residual in out_res_map.items():
            op.declare_state(output, residual=residual)
        op.nonlinear_solver = nonlinear_solver
        op.linear_solver = linear_solver
        inputs = [
            self.declare_variable(input_name, shape=input_shape)
            for input_name, input_shape in input_data.items()
        ]
        outputs = op.apply(*inputs)


class HybridImplicitOperation(CustomExplicitOperation):

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
        # allow Output types for exposed intermediate variables
        exposed_residuals: Set[str],
        *args,
        expose: List[str] = [],
        defaults: Dict[str, np.ndarray] = dict(),
        nonlinear_solver: Union[NonlinearSolver, None] = None,
        linear_solver: Union[LinearSolver, None] = None,
        backend: str = '',
        **kwargs,
    ):
        self.rep = rep
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
        self.res_out_map: Dict[str, DeclaredVariable] = res_out_map
        self.out_res_map: Dict[str, Output] = out_res_map
        self.out_in_map: Dict[str, List[DeclaredVariable]] = out_in_map
        self.exp_in_map: Dict[str, List[DeclaredVariable]] = exp_in_map
        self.expose: List[str] = expose
        self.exposed_residuals: Set[str] = exposed_residuals
        self.nonlinear_solver: Union[NonlinearSolver,
                                     None] = nonlinear_solver
        self.linear_solver: Union[LinearSolver, None] = linear_solver
        self.defaults = defaults
        self.exposed_variables: Dict[str, Output] = exposed_variables

        self.name: str = name
        self.mode: str = mode
        self.model: 'Model' = model

        self._build_internal_simulator()

        # inputs are always inputs
        self.input_data: Set[Tuple[str, Tuple[int, ...]]] = set()
        for inputs in out_in_map.values():
            for inp in inputs:
                pair = (inp.name, inp.shape)
                self.input_data.add(pair)
        self.output_data: Set[Tuple[str, Tuple[int, ...]]] = set()
        if self.mode == 'implicit':
            # implicit outputs are outputs when converging residuals
            for output in res_out_map.values():
                pair = (output.name, output.shape)
                self.output_data.add(pair)
        if self.mode == 'explicit':
            # implicit outputs are inputs when computing residuals
            # explicitly
            for output in res_out_map.values():
                pair = (output.name, output.shape)
                self.input_data.add(pair)

            # exposed explicit outputs are included in ouputs when
            # computing residuals explicitly
            for n, v in self.exposed_variables.items():
                self.output_data.add((n, v.shape))

            # residuals are outputs when computing residuals
            # explicitly
            self.residual_data: Set[Tuple[str, Tuple[int, ...]]] = set()
            for residual in out_res_map.values():
                pair = (residual.name, residual.shape)
                self.output_data.add(pair)

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
                    nonlinear_solver=self.nonlinear_solver,
                    linear_solver=self.linear_solver,
                ), )
            self.sim = Simulator(rep)

    def define(self):
        self.totals = dict()
        self.outputs = dict()
        self.inputs = dict()
        self.residuals = dict()

        for (n, s) in self.input_data:
            self.add_input(n, shape=s)
        if self.mode == 'implicit':
            for (n, s) in self.output_data:
                self.add_output(n, shape=s)
        elif self.mode == 'explicit':
            for (n, s) in self.output_data:
                self.add_input(n, shape=s)
            for (n, s) in self.residual_data:
                self.add_output(n, shape=s)

        if self.mode == 'explicit':
            self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):
        for name in self.input_meta.keys():
            self.sim[name] = inputs[name]

        self.sim.run()

        for name in self.output_meta.keys():
            outputs[name] = self.sim[name]

    def compute_derivatives(self, inputs, derivatives):
        if self.mode == 'explicit':
            input_names: List[str] = list(self.input_meta.keys())
            output_names: List[str] = list(self.output_meta.keys())

            self.totals = self.sim.compute_totals(output_names,
                                                  input_names)

            for of in output_names:
                for wrt in input_names:
                    derivatives[of, wrt] = self.totals[of, wrt]

    # Some functions that may make things easier for back end developers
    def inputs(self):
        return {k: self.sim[k] for k in list(self.input_meta.keys())}

    def outputs(self):
        return {k: self.sim[k] for k in list(self.output_meta.keys())}

    def partial_derivatives(self):
        return self.totals

    def residuals(self):
        return self.residuals

    def set_tolerance(self, tol: float):
        self.nonlinear_solver.options['atol'] = tol
