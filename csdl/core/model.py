from contextlib import contextmanager
from typing import Callable, Tuple, List, Dict
from copy import deepcopy
from csdl.utils.typehints import Shape

import networkx as nx
from csdl.core.intermediate_representation import IntermediateRepresentation
from csdl.core.implicit_operation import ImplicitOperation
from csdl.core.bracketed_search_operation import BracketedSearchOperation
from csdl.core.node import Node
from csdl.core.variable import Variable
from csdl.core.declared_variable import DeclaredVariable
from csdl.core.input import Input
from csdl.core.output import Output
from csdl.core.concatenation import Concatenation
from csdl.core.custom_operation import CustomOperation
from csdl.core.subgraph import Subgraph
from csdl.core.implicit_operation_factory import ImplicitOperationFactory
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.linear_solver import LinearSolver
from csdl.operations.print_var import print_var
from csdl.utils.parameters import Parameters
from csdl.utils.collect_terminals import collect_terminals
from csdl.utils.check_default_val_type import check_default_val_type
from csdl.utils.check_constraint_value_type import check_constraint_value_type
from warnings import warn
import numpy as np
from csdl.core.compiler_middle_end import CompilerFrontEndAndMiddleEnd

_residual = '_residual'


class Model(metaclass=CompilerFrontEndAndMiddleEnd):
    _count = -1

    def __init__(self, **kwargs):
        Model._count += 1
        self._main = True
        self._defined = False
        self.symbol_table: dict = {}
        self.sorted_nodes: List[Node] = []
        self._most_recently_added_subgraph: Subgraph = None
        self.subgraphs: List[Subgraph] = []
        self.variables_promoted_from_children: List[Variable] = []
        self.inputs: List[Input] = []
        self.declared_variables: List[DeclaredVariable] = []
        self.registered_outputs: List[Output | Subgraph] = []
        self.objective: dict | None = None
        self.constraints: Dict[str, dict] = dict()
        self.design_variables: Dict[str, dict] = dict()
        self.user_declared_connections: List[Tuple[str, str]] = []
        self.connections: List[Tuple[str, str]] = []
        self.parameters: Parameters = Parameters()
        self.initialize()
        self.parameters.update(kwargs)
        self._optimize_ir = False
        self.sources_promoted_from_submodels: Dict[str, Shape] = dict()
        """
        Map from name to shape for sources (inputs and outputs)
        promoted from submodels
        """
        self.sinks_promoted_from_submodels: Dict[str, Shape] = dict()
        """
        Map from name to shape for sinks (declared variables)
        promoted from submodels
        """

    def optimize_ir(self, flag: bool = True):
        self._optimize_ir = flag

    def initialize(self):
        """
        User defined method to declare parameter values.
        Parameters are compile time constants (neither inputs nor
        outputs to the model) and cannot be updated at runtime.
        Parameters are intended to make a `Model` subclass definition
        generic, and therefore reusable.
        The example below shows how a `Model` subclass definition uses
        parameters and how the user can set parameters when constructing
        the example `Model` subclass.

        **Example**


        ```py
        class Example(Model):
            def initialize(self):
                self.parameters.declare('num_times', types=int)
                self.parameters.declare('step_size', types=float)
                self.parameters.declare('surface', types=dict)

            def define(self):
                num_times = self.parameters['num_times']
                step_size = self.parameters['step_size']
                surface = self.parameters['surface']
                name = surface['name'] # str
                symmetry = surface['symmetry'] # bool
                mesh = surface['mesh'] # numpy array

                # define runtime behavior...

            surface = {
                'name': 'wing',
                'symmetry': False,
                'mesh': mesh,
            }

            # compile using Simulator imported from back end...
            sim = Simulator(
                Example(
                    num_times=100,
                    step_size=0.1,
                    surface=surface,
                ),
            )
        ```

        """
        pass

    def define(self):
        """
        User defined method to define runtime behavior.
        Note: the user never _calls_ this method. Only the `Simulator`
        class constructor calls this method.

        **Example**

        ```py
        class Example(Model):
            def define(self):
                self.create_input('x')
                m = 5
                b = 3
                y = m*x + b
                self.register_output('y', y)

        # compile using Simulator imported from back end...
        sim = Simulator(Example())
        sim['x'] = -3/5
        sim.run()
        print(sim['y']) # expect 0
        ```
        """
        pass

    def _redefine(self, expose: List[str]):
        """
        Remove edges so that we can update the graph.
        This is *only* used when exposing intermediate variables for a
        composite residual.
        """
        if len(expose) > 0:
            if self._defined == True:
                self._defined = False
                for out in self.registered_outputs:
                    out.remove_fwd_edges()
                for out in self.registered_outputs:
                    out.remove_dependencies()
            from csdl.operations.passthrough import passthrough
            se = set(expose)
            vars = list(
                filter(lambda x: x.name in se, self.registered_outputs))
            for var in vars:
                op = passthrough(var)
                out = Output(
                    _residual + var._id,
                    val=var.val,
                    shape=var.shape,
                    units=var.units,
                    desc=var.desc,
                    tags=var.tags,
                    shape_by_conn=var.shape_by_conn,
                    copy_shape=var.copy_shape,
                    distributed=var.distributed,
                    op=op,
                )
                op.outs = (out, )
                self.register_output(out.name, out)
        self.define()

    def print_var(self, var: Variable):
        """
        Print *runtime* value during execution. Note that `print_var`
        should only be used for debugging, as it does have a performance
        impact. Note that Python's `print` function will print the
        CSDL compile time `Variable` object information, and will have
        no effect on run time execution.

        **Example**

        ```py
        y = csdl.sin(x)
        print(y) # will print compile time information about y
        self.print_var(y) # will print run time value of y
        ```
        """
        if not isinstance(var, Variable):
            raise TypeError(
                "CSDL can only print information about Variable objects"
            )
        op = print_var(var)
        out = Output(
            var.name + '_print',
            val=var.val,
            shape=var.shape,
            units=var.units,
            desc=var.desc,
            tags=var.tags,
            shape_by_conn=var.shape_by_conn,
            copy_shape=var.copy_shape,
            distributed=var.distributed,
            op=op,
        )
        self.register_output(out.name, out)

    def add_objective(
        self,
        name,
        ref=None,
        ref0=None,
        index=None,
        units=None,
        adder=None,
        scaler=None,
        parallel_deriv_color=None,
        cache_linear_solution=False,
    ):
        """
        Declare the objective for the optimization problem. Objective
        must be a scalar variable.
        """
        self.objective = dict(
            name=name,
            ref=ref,
            ref0=ref0,
            index=index,
            units=units,
            adder=adder,
            scaler=scaler,
            parallel_deriv_color=parallel_deriv_color,
            cache_linear_solution=cache_linear_solution,
        )

    def add_design_variable(
        self,
        name,
        lower=None,
        upper=None,
        ref=None,
        ref0=None,
        indices=None,
        adder=None,
        scaler=None,
        units=None,
        parallel_deriv_color=None,
        cache_linear_solution=False,
    ):
        """
        Add a design variable to the optimization problem. The design
        variable must be an `Input`. This will signal to the optimizer
        that it is responsible for updating the input variable.
        """
        if name in self.design_variables.keys():
            raise ValueError(
                "{} already added as a design variable".format(name))
        self.design_variables[name] = dict(
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            indices=indices,
            adder=adder,
            scaler=scaler,
            units=units,
            parallel_deriv_color=parallel_deriv_color,
            cache_linear_solution=cache_linear_solution,
        )

    def add_constraint(
        self,
        name,
        lower=None,
        upper=None,
        equals=None,
        ref=None,
        ref0=None,
        adder=None,
        scaler=None,
        units=None,
        indices=None,
        linear=False,
        parallel_deriv_color=None,
        cache_linear_solution=False,
    ):
        """
        Add a constraint to the optimization problem.
        """
        if name in self.constraints.keys():
            raise ValueError(
                "Constraint already defined for {}".format(name))
        else:
            check_constraint_value_type(lower)
            check_constraint_value_type(upper)
            check_constraint_value_type(equals)

            if lower is not None and upper is not None:
                if np.greater(lower, upper):
                    raise ValueError(
                        "Lower bound is greater than upper bound:\n lower bound: {}\n upper bound: {}"
                        .format(lower, upper))
            self.constraints[name] = dict(
                lower=lower,
                upper=upper,
                equals=equals,
                ref=ref,
                ref0=ref0,
                adder=adder,
                scaler=scaler,
                units=units,
                indices=indices,
                linear=linear,
                parallel_deriv_color=parallel_deriv_color,
                cache_linear_solution=cache_linear_solution,
            )

    def connect(self, a: str, b: str):
        warn(
            "Error messages for connections are not yet built into "
            "CSDL frontend. Pay attention to any errors emmited by back end."
        )
        if (a, b) in self.user_declared_connections:
            warn("Connection from {} to {} issued twice.".format(a, b))
        else:
            self.user_declared_connections.append((a, b))

    def declare_variable(
        self,
        name: str,
        val=1.0,
        shape=(1, ),
        src_indices=None,
        flat_src_indices=None,
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
    ) -> DeclaredVariable:
        """
        Declare an input to use in an expression.

        An input can be an output of a child `System`. If the user
        declares an input that is computed by a child `System`, then
        the call to `self.declare_variable` must appear after the call to
        `self.add`.

        **Parameters**

        name: str
            Name of variable in CSDL to be used as a local input that
            takes a value from a parent model, child model, or
            previously registered output within the model.
        shape: Tuple[int]
            Shape of variable
        val: Number or ndarray
            Default value for variable

        **Returns**

        DocInput
            An object to use in expressions
        """
        v = DeclaredVariable(
            name,
            val=check_default_val_type(val),
            shape=shape,
            src_indices=src_indices,
            flat_src_indices=flat_src_indices,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            distributed=distributed,
        )
        self.declared_variables.append(v)
        return v

    def create_input(
        self,
        name,
        val=1.0,
        shape=(1, ),
        units=None,
        desc='',
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
    ) -> Input:
        """
        Create an input to the main model, whose value remains constant
        during model evaluation.

        **Parameters**

        name: str
            Name of variable in CSDL
        shape: Tuple[int]
            Shape of variable
        val: Number or ndarray
            Value for variable during first model evaluation

        **Returns**

        Input
            An object to use in expressions
        """
        i = Input(
            name,
            val=check_default_val_type(val),
            shape=shape,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            distributed=distributed,
        )
        self.inputs.append(i)
        return i

    def create_output(
        self,
        name,
        val=1.0,
        shape=(1, ),
        units=None,
        res_units=None,
        desc='',
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=1.0,
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        distributed=None,
    ) -> Concatenation:
        """
        Create a value that is computed explicitly, either through
        indexed assignment, or as a fixed point iteration.

        **Example**

        ```py
        x = self.create_output('x', shape=(5,3,2))
        x[:, :, 0] = a
        x[:, :, 1] = b
        ```

        **Parameters**

        name: str
            Name of variable in CSDL
        shape: Tuple[int]
            Shape of variable

        **Returns**

        Concatenation
            An object to use in expressions
        """
        c = Concatenation(
            name,
            val=check_default_val_type(val),
            shape=shape,
            units=units,
            desc=desc,
            tags=tags,
            shape_by_conn=shape_by_conn,
            copy_shape=copy_shape,
            res_units=res_units,
            lower=lower,
            upper=upper,
            ref=ref,
            ref0=ref0,
            res_ref=res_ref,
            distributed=distributed,
        )
        self.register_output(name, c)

    def register_output(self, name: str, var: Output) -> Output:
        """
        Register `var` as an output of the `Model`.
        When adding subsystems, each of the submodel's inputs requires
        a call to `register_output` prior to the call to
        `add`.

        **Parameters**

        `name: str`

            Name of variable in CSDL

        `var: Output`

            Variable that defines output

        **Returns**

        `Output`

            Variable that defines output (same object as argument)
        """
        if not isinstance(var, Output):
            raise TypeError(
                'Can only register Output object as an output. Received type {}.'
                .format(type(var)))
        else:
            if var in self.registered_outputs:
                raise ValueError(
                    "Cannot register output twice; attempting to register "
                    "{} as {}.".format(var.name, name))

            var.name = name
            self.registered_outputs.append(var)
        return var

    # TODO: what to do about create_model?
    # TODO: never add subgraph to registered_outputs
    # TODO: check for duplicate subgraph names
    # TODO: establish dependencies based on promotions and connections
    # (middle end)
    def add(
        self,
        submodel,
        name: str | None = None,
        promotes: List[str] | None = None,
    ) -> 'Model':
        """
        Add a submodel to the `Model`.

        `self.add` call must be preceded by a call to
        `self.register_output` for each of the submodel's inputs,
        and followed by `self.declare_variable` for each of the
        submodel's outputs.

        **Parameters**

        name: str
            Name of submodel
        submodel: System
            Subsystem to add to `Model`
        promotes: List
            Variables to promote

        **Returns**

        System
            Subsystem to add to `Model`
        """
        if not isinstance(submodel, Model):
            raise TypeError("{} is not a Model".format(submodel))

        # Prevent certain compile time functions from executing for any
        # model other than the main model
        submodel._main = False

        subgraph = Subgraph(
            name,
            submodel,
            promotes=promotes,
        )
        self.subgraphs.append(subgraph)

        return submodel

    def _bracketed_search(
        self,
        implicit_metadata: Dict[str, dict],
        *arguments: Variable,
        states: List[str],
        residuals: List[str],
        model,
        brackets: Dict[str, Tuple[int | float | np.ndarray,
                                  int | float | np.ndarray]],
        expose: List[str] = [],
        maxiter: int = 100,
    ):
        """
        Create an implicit operation whose residuals are defined by a
        `Model`.
        An implicit operation is an operation that solves an equation
        $f(x,y)=0$ for $y$, given some value of $x$.
        CSDL solves $f(x,y)=0$ by defining a residual $r=f(x,y)$ and
        updating $y$ until $r$ converges to zero.

        **Parameters**

        `arguments: List[Variable]`

        > List of variables to use as arguments for the implicit
        > operation.
        > Variables must have the same name as a declared variable
        > within the `model`'s class definition.

        :::note
        The declared variable _must_ be declared within `model`
        _and not_ promoted from a child submodel.
        :::

        `states: List[str]`

        > Names of states to compute using the implicit operation.
        > The order of the names of these states corresponds to the
        > order of the output variables returned by
        > `implicit_operation`.
        > The order of the names in `states` must also match the order
        > of the names of the residuals associated with each state in
        > `residuals`.

        :::note
        The declared variable _must_ be declared within `model`
        _and not_ promoted from a child submodel.
        :::

        `residuals: List[str]`

        > The residuals associated with the states.
        > The name of each residual must match the name of a
        > registered output in `model`.

        :::note
        The registered output _must_ be registered within `model`
        _and not_ promoted from a child submodel.
        :::

        `model: Model`

        > The `Model` object to use to define the residuals.
        > Residuals may be defined via functional composition and/or
        > hierarchical composition.

        :::note
        _Any_ `Model` object may be used to define residuals for an
        implicit operation
        :::

        `nonlinear_solver: NonlinearSolver`

        > The nonlinear solver to use to converge the residuals

        `linear_solver: LinearSolver`

        > The linear solver to use to solve the linear system

        `expose: List[str]`

        > List of intermediate variables inside `model` that are
        > required for computing residuals to which it is desirable
        > to have access outside of the implicit operation.
        > For example, if a trajectory is computed using time marching
        > and a residual is computed from the final state of the
        > trajectory, it may be desirable to plot that trajectory
        > after the conclusion of a simulation, e.g. after an
        > iteration during an optimization process.

        :::note
        The variable names in `expose` may be any name within the
        model hierarchy defined in `model`, but the variable names
        in `expose` are neither declared variables, nor registered
        outputs in `model`, although they may be declared
        variables/registered outputs in a submodel (i.e. they are
        neither states nor residuals in the, implicit operation).
        :::

        **Returns**

        `Tuple[Ouput]`

        > Variables to use in this `Model`.
        > The variables are named according to `states` and `expose`,
        > and are returned in the same order in which they are
        > declared.
        > For example, if `states=['a', 'b', 'c']` and
        > `expose=['d', 'e', 'f']`, then the outputs
        > `a, b, c, d, e, f` in
        > `a, b, c, d, e, f = self.implcit_output(...)`
        > will be named
        > `'a', 'b', 'c', 'd', 'e', 'f'`, respectively.
        > This enables use of exposed intermediate variables (in
        > addition to the states computed by converging the
        > residuals) from `model` in this `Model`.
        > Unused outputs will be ignored, so
        > `a, b, c = self.implcit_output(...)`
        > will make the variables declared in `expose` available for
        > recording/analysis and promotion/connection, but they will
        > be unused by this `Model`.
        > Note that these variables are already registered as outputs
        > in this `Model`, so there is no need to call
        > `Model.register_output` for any of these variables.
        """
        out_res_map, res_out_map, out_in_map = self._something(
            model,
            arguments,
            states,
            residuals,
            expose,
        )

        new_brackets: Dict[str, Tuple[np.ndarray, np.ndarray]] = dict()
        states_without_brackets = deepcopy(states)
        for k, v in brackets.items():
            if k not in states:
                raise ValueError(
                    "No state {} for specified bracket {}".format(k, v))
            if k in states_without_brackets:
                states_without_brackets.remove(k)

            if len(v) != 2:
                raise ValueError(
                    "Bracket {} for state {} is not a tuple of two values"
                    .format(v, k))

            (a, b) = (np.array(v[0]), np.array(v[1]))
            if a.shape != b.shape:
                raise ValueError(
                    "Bracket values for {} are not the same shape; {} != {}"
                    .format(k, a.shape, b.shape))
            new_brackets[k] = (a, b)

        if len(states_without_brackets) > 0:
            raise ValueError(
                "The following states are missing brackets: {}".format(
                    states_without_brackets))

        op = BracketedSearchOperation(
            model=model,
            out_res_map=out_res_map,
            res_out_map=res_out_map,
            out_in_map=out_in_map,
            brackets=new_brackets,
            maxiter=maxiter,
            expose=expose,
        )

        return self._return_implicit_outputs(
            model,
            op,
            arguments,
            states,
            residuals,
            expose,
            implicit_metadata,
        )

    def _implicit_operation(
        self,
        implicit_metadata: Dict[str, dict],
        *arguments: Variable,
        states: List[str],
        residuals: List[str],
        model: 'Model',
        nonlinear_solver: NonlinearSolver,
        linear_solver: LinearSolver | None = None,
        expose: List[str] = [],
        defaults: Dict[str, int | float | np.ndarray] = dict(),
    ) -> Tuple[Output, ...]:
        """
        Create an implicit operation whose residuals are defined by a
        `Model`.
        An implicit operation is an operation that solves an equation
        $f(x,y)=0$ for $y$, given some value of $x$.
        CSDL solves $f(x,y)=0$ by defining a residual $r=f(x,y)$ and
        updating $y$ until $r$ converges to zero.

        **Parameters**

        `arguments: List[Variable]`

            List of variables to use as arguments for the implicit
            operation.
            Variables must have the same name as a declared variable
            within the `model`'s class definition.

            :::note
            The declared variable _must_ be declared within `model`
            _and not_ promoted from a child submodel.
            :::

        `states: List[str]`

            Names of states to compute using the implicit operation.
            The order of the names of these states corresponds to the
            order of the output variables returned by
            `implicit_operation`.
            The order of the names in `states` must also match the order
            of the names of the residuals associated with each state in
            `residuals`.

            :::note
            The declared variable _must_ be declared within `model`
            _and not_ promoted from a child submodel.
            :::

        `residuals: List[str]`

            The residuals associated with the states.
            The name of each residual must match the name of a
            registered output in `model`.

            :::note
            The registered output _must_ be registered within `model`
            _and not_ promoted from a child submodel.
            :::

        `model: Model`

            The `Model` object to use to define the residuals.
            Residuals may be defined via functional composition and/or
            hierarchical composition.

            :::note
            _Any_ `Model` object may be used to define residuals for an
            implicit operation
            :::

        `nonlinear_solver: NonlinearSolver`

            The nonlinear solver to use to converge the residuals

        `linear_solver: LinearSolver`

            The linear solver to use to solve the linear system

        `expose: List[str]`

            List of intermediate variables inside `model` that are
            required for computing residuals to which it is desirable
            to have access outside of the implicit operation.

            For example, if a trajectory is computed using time marching
            and a residual is computed from the final state of the
            trajectory, it may be desirable to plot that trajectory
            after the conclusion of a simulation, e.g. after an
            iteration during an optimization process.

            :::note
            The variable names in `expose` may be any name within the
            model hierarchy defined in `model`, but the variable names
            in `expose` are neither declared variables, nor registered
            outputs in `model`, although they may be declared
            variables/registered outputs in a submodel (i.e. they are
            neither states nor residuals in the, implicit operation).
            :::

        **Returns**

        `Tuple[Ouput]`

            Variables to use in this `Model`.
            The variables are named according to `states` and `expose`,
            and are returned in the same order in which they are
            declared.
            For example, if `states=['a', 'b', 'c']` and
            `expose=['d', 'e', 'f']`, then the outputs
            `a, b, c, d, e, f` in
            `a, b, c, d, e, f = self.implcit_output(...)`
            will be named
            `'a', 'b', 'c', 'd', 'e', 'f'`, respectively.
            This enables use of exposed intermediate variables (in
            addition to the states computed by converging the
            residuals) from `model` in this `Model`.
            Unused outputs will be ignored, so
            `a, b, c = self.implcit_output(...)`
            will make the variables declared in `expose` available for
            recording/analysis and promotion/connection, but they will
            be unused by this `Model`.
            Note that these variables are already registered as outputs
            in this `Model`, so there is no need to call
            `Model.register_output` for any of these variables.
        """
        out_res_map, res_out_map, out_in_map = self._something(
            model,
            arguments,
            states,
            residuals,
            expose,
        )
        new_default_values: Dict[str, np.ndarray] = dict()
        for k, v in defaults.items():
            if k not in states:
                raise ValueError(
                    "No state {} for specified default value {}".format(
                        k, v))
            if not isinstance(v, (int, float, np.ndarray)):
                raise ValueError(
                    "Default value for state {} is not an int, float, or ndarray"
                    .format(k))
            if isinstance(v, np.ndarray):
                f = list(
                    filter(lambda x: x.name == k,
                           model.registered_outputs))
                if len(f) > 0:
                    if f[0].shape != v.shape:
                        raise ValueError(
                            "Shape of value must match shape of state {}; {} != {}"
                            .format(k, f[0].shape, v.shape))
                new_default_values[k] = np.array(v) * np.ones(
                    f[0].shape)

        # create operation, establish dependencies on arguments
        op = ImplicitOperation(
            model=model,
            nonlinear_solver=nonlinear_solver,
            linear_solver=linear_solver,
            out_res_map=out_res_map,
            res_out_map=res_out_map,
            out_in_map=out_in_map,
            expose=expose,
            defaults=new_default_values,
        )

        return self._return_implicit_outputs(
            model,
            op,
            arguments,
            states,
            residuals,
            expose,
            implicit_metadata,
        )

    def _something(
        self,
        model,
        arguments,
        states,
        residuals,
        expose: List[str] = [],
    ):
        # TODO: check if simulation time is deterministic; depending on
        # how some std operations are implemented, it may not be true to
        # say that deterministic => no
        # ImplicitOperation/BracketedSearchOperation
        # self.deterministic = False
        if not isinstance(model, (Model, )):
            raise TypeError("{} is not a Model".format(model))
        # if model._defined is False:
        #     model.define()

        # check for duplicate arguments, states, and residuals
        arg_names = [var.name for var in arguments]
        if len(set(arg_names)) < len(arg_names):
            raise ValueError("Duplicate arguments found")
        if len(set(states)) < len(states):
            raise ValueError("Duplicate names for states found")
        if len(set(residuals)) < len(residuals):
            raise ValueError("Duplicate names for residuals found")

        if len(states) != len(residuals):
            raise ValueError(
                "Number of states and residuals must be equal")

        # We need to have access to lists of declared variables and
        # registered outputs, so we run the compiler front end and
        # middle end for the internal model that defines the residuals;
        # but first, we need to make sure that if the user has called
        # model.define(), then it will have no effect; we need to do
        # this because we are going to register new outputs, the
        # residuals associated with the exposed variables
        model._redefine(expose)

        # NOTE: top level inputs will be unused, so we don't allow them
        if len(model.inputs) > 0:
            raise ValueError(
                "The model that defines residuals is not allowed to"
                "define top level inputs (i.e. calls to"
                "`Model.create_input`).")

        # ignore optimization problem defined in model, if any;
        # this instance of the model is not defining an optimization
        # problem
        model.design_variables = dict()
        model.objective = None
        model.constraints = dict()

        # After this runs, the internal model only computes
        # residuals and exposed outputs
        # NOTE: model.sorted_nodes is unaffected; remove_unused_outputs
        # will have no effect on performance
        # remove_unused_outputs(
        #     model,
        #     residuals,
        #     expose,
        # )

        # TODO: expose residuals automatically (SURF)

        # TODO: transfer metadata to declared variables in model
        # check that name and shape of each argument matches name and
        # shape of a declared variable in internal model
        for arg in arguments:
            arg_name_match = False
            for var in model.declared_variables:
                if arg.name == var.name:
                    arg_name_match = True
                    if arg.shape != var.shape:
                        raise ValueError(
                            "The argumet {} has shape {}, which does not match the shape {} of the declared variable of the model used to define an implicit operation"
                            .format(arg.name, arg.shape, var.shape))
                    var.val = arg.val
            if arg_name_match is False:
                raise ValueError(
                    "The argument {} is not a declared variable of the model used to define an implicit operation"
                    .format(arg.name))

        # check that name of each state matches name of a declared
        # variable in internal model
        for state_name in states:
            state_name_match = False
            for var in model.declared_variables:
                if state_name == var.name:
                    state_name_match = True
                    # TODO: initialize state value from outside the model
            if not state_name_match:
                raise ValueError(
                    "The state {} is not a declared variable of the model used to define an implicit operation"
                    .format(state_name))

        # check that name of each residual matches name of a registered
        # output in internal model
        for residual_name in residuals:
            residual_name_match = False
            for var in model.registered_outputs:
                if residual_name == var.name:
                    residual_name_match = True
            if not residual_name_match:
                raise ValueError(
                    "The residual {} is not a registered output of the model used to define an implicit operation"
                    .format(residual_name))

        # TODO: this needs to be more involved because user is allowed to
        # expose any variable in the hierarchy
        # check that name of each exposed intermediate output matches
        # name of a registered output in internal model
        for expose_name in expose:
            expose_name_match = False
            for var in model.registered_outputs:
                if expose_name == var.name:
                    expose_name_match = True
            if not expose_name_match:
                raise ValueError(
                    "The intermediate output {} is not a registered output of the model used to define an implicit operation"
                    .format(expose_name))

        # TODO: always expose residuals (SURF)

        # make some maps to do some things more efficiently later
        registered_output_map: Dict[str, Output] = dict()
        declared_variables_map: Dict[str, DeclaredVariable] = dict()
        exposed_residuals_map: Dict[str, Output] = dict()
        res_exp_map: Dict[str, Output] = dict()
        exp_res_map: Dict[str, Output] = dict()

        for res in model.registered_outputs:
            if res.name.startswith(_residual):
                # residual is associated with exposed variable;
                # map exposed variables to exposed residuals and vice
                # versa
                exposed_residuals_map[res.name] = res
                expose_var = res.dependencies[0].dependencies[0]
                res_exp_map[res.name] = expose_var
                expose_name = expose_var.name
                exp_res_map[expose_name] = res
            elif res.name not in expose:
                registered_output_map[res.name] = res

        for dv in model.declared_variables:
            declared_variables_map[dv.name] = dv

        # create two-way mapping between state objects and residual
        # objects
        # TODO: explain why
        out_res_map: Dict[str, Output] = dict()
        res_out_map: Dict[str, DeclaredVariable | Output] = dict()
        for s, r in zip(states, residuals):
            if registered_output_map[r].shape != declared_variables_map[
                    s].shape:
                raise ValueError(
                    "Shape of state {} and residual {} do not match.".
                    format(s, r))
            out_res_map[s] = registered_output_map[r]
            res_out_map[r] = declared_variables_map[s]
        # merge dictionaries for residuals associated with states and
        # residuals associated with explicitly computed, exposed,
        # intermediate values
        res_out_map = {**res_out_map, **res_exp_map}
        out_res_map = {**out_res_map, **exp_res_map}

        # TODO: (?) keep track of which exposed variables depend on
        # residuals; necessary for computing derivatives of residuals
        # associated with exposed variables wrt exposed variables, but
        # only those exposed variables that do not depend on a stata

        # TODO: collect_terminals for exposed variables, like an "exp_in_map"

        # TODO: check if residuals depend on each other
        # Associate outputs with the inputs they depend on
        # Collect residual expressions and their corresponding inputs
        # and outputs
        out_in_map: Dict[str, List[DeclaredVariable]] = dict()
        for state_name, residual in out_res_map.items():
            # Collect inputs (terminal nodes) for this residual only; no
            # duplicates
            in_vars = list(
                set(collect_terminals(
                    [],
                    residual,
                    residual,
                )))

            if state_name in states and state_name not in [
                    var.name for var in in_vars
            ]:
                raise ValueError(
                    "Residual {} does not depend on state {}".format(
                        residual.name, state_name))

            # Store inputs for this implicit output
            # Only the arguments specified in the parent model will be
            # inputs to the internal model
            argument_names = [x.name for x in arguments]
            out_in_map[state_name] = list(
                filter(lambda v: v.name in argument_names + states,
                       in_vars))
            # out_in_map[state_name] = in_vars

        for state_name, residual in exp_res_map.items():
            # Collect inputs (terminal nodes) for this residual only; no
            # duplicates
            in_vars = list(
                set(collect_terminals(
                    [],
                    residual,
                    residual,
                )))

            # Store inputs for this implicit output
            out_in_map[state_name] = in_vars

        # # check that exposed variable are valid choices
        # # use model.sorted_nodes
        # f = filter(lambda x: isinstance(x, Variable),
        #            model.sorted_nodes)
        # # collect all variables in model hierarchy
        # all_variable_names = [var.name for var in f]
        # all_variable_names.extend(
        #     model.variables_promoted_from_children)

        # # remove top level declared variables, registered outputs
        # # these are arguments, states, and residuals
        # names_of_variables_not_to_expose = set([
        #     var.name for var in list(
        #         set(model.declared_variables).union(
        #             set(model.registered_outputs)))
        # ])
        # intermediate_variable_names = set(
        #     all_variable_names).difference(
        #         names_of_variables_not_to_expose)

        # # check that variables user requests to expose are defined
        # for name in expose:
        #     if name not in intermediate_variable_names:
        #         raise ValueError(
        #             "Cannot expose {} because it is not an intermediate variable"
        #             .format(name))
        return (out_res_map, res_out_map, out_in_map)

    def _return_implicit_outputs(
        self,
        model,
        op: ImplicitOperation,
        arguments: Tuple[Variable, ...],
        states: List[str],
        residuals: List[str],
        expose: List[str],
        implicit_metadata: Dict[str, dict],
    ) -> Tuple[Output, ...]:
        for arg in arguments:
            op.add_dependency_node(arg)

        # create outputs of operation, establish dependencies on
        # operation, and register outputs
        outs: List[Output] = []
        for s, r in zip(states, residuals):

            internal_var = list(
                filter(lambda x: x.name == s,
                       model.declared_variables))[0]

            out = Output(
                s,
                shape=internal_var.shape,
                **implicit_metadata[s],
                op=op,
            )
            self.register_output(s, out)
            outs.append(out)

        # include exposed intermediate outputs in IR
        se = set(expose)
        ex = filter(lambda x: x.name in se, model.registered_outputs)
        for e in ex:
            out = Output(
                e.name,
                val=e.val,
                shape=e.shape,
                units=e.units,
                desc=e.desc,
                tags=e.tags,
                shape_by_conn=e.shape_by_conn,
                copy_shape=e.copy_shape,
                distributed=e.distributed,
                op=op,
            )
            self.register_output(e.name, out)
            outs.append(out)

        # ensure operation has knowledge of outputs so that back end can
        # generate code from operation
        op.outs = outs

        # return outputs
        if len(outs) > 1:
            return tuple(outs)
        else:
            return outs[0]

    def create_implicit_operation(self, model):
        model.optimize_ir(self._optimize_ir)
        return ImplicitOperationFactory(self, model)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.define()

    @contextmanager
    def create_submodel(self, name: str):
        """
        Create a `Model` object and add as a submodel, promoting all
        inputs and outputs.
        For use in `with` contexts.
        NOTE: Only use if planning to promote all varaibales within
        child `Model` object.

        **Parameters**

        name: str
            Name of new child `Model` object

        **Returns**

        Model
            Child `Model` object whose variables are all promoted
        """
        try:
            with Model() as m:
                yield m
        finally:
            self.add(m, name=name)
