from contextlib import contextmanager
from re import X
from typing import Any, Dict, List, Set, Tuple, Union
from copy import copy
from csdl.utils.typehints import Shape
from csdl.utils.get_shape_val import get_shape_val

from csdl.rep.graph_representation import GraphRepresentation
from csdl.lang.implicit_operation import ImplicitOperation
from csdl.lang.bracketed_search_operation import BracketedSearchOperation
from csdl.lang.node import Node
from csdl.lang.variable import Variable
from csdl.lang.declared_variable import DeclaredVariable
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.concatenation import Concatenation
from csdl.lang.subgraph import Subgraph
from csdl.lang.implicit_operation_factory import ImplicitOperationFactory
from csdl.solvers.nonlinear_solver import NonlinearSolver
from csdl.solvers.linear_solver import LinearSolver
from csdl.operations.print_var import print_var
from csdl.utils.parameters import Parameters
from csdl.utils.collect_terminals import collect_terminals
from csdl.utils.check_default_val_type import check_default_val_type
from csdl.utils.check_constraint_value_type import check_constraint_value_type
from warnings import warn
import numpy as np
try:
    from csdl.operations.passthrough import passthrough
except ImportError:
    pass
from collections import OrderedDict

# TODO: if defined, raise error on each user facing method


def _to_array(
    x: Union[int, float, np.ndarray, Variable]
) -> Union[np.ndarray, Variable]:
    if not isinstance(x, (np.ndarray, Variable)):
        x = np.array(x)
    return x


class Model:
    _count = -1

    def __init__(self, **kwargs):
        Model._count += 1
        self.defined = False
        self.subgraphs: List[Subgraph] = []
        self.variables_promoted_from_children: List[Variable] = []

        self.inputs: List[Input] = []
        self.inputs_set: Set[Input] = set()
        self.input_names_set: Set[Input] = set()

        self.declared_variables: List[DeclaredVariable] = []
        self.declared_variables_set: Set[DeclaredVariable] = set()
        self.declared_variable_names_set: Set[DeclaredVariable] = set()

        self.registered_outputs: List[Output] = []
        self.registered_outputs_set: Set[Output] = set()
        self.registered_output_names_set: Set[Output] = set()

        self.objective: Dict[str, Any] = dict()
        self.constraints: Dict[str, Dict[str, Any]] = dict()
        self.design_variables: Dict[str, Dict[str, Any]] = dict()
        self.user_declared_connections: List[Tuple[str, str]] = []
        """
        User specified connections, stored as they appear in user code.
        Both relative unpromoted names and unique promoted names are
        allowed.
        """
        self.connections: List[Tuple[str, str]] = []
        """
        User specified connections, stored using unique, promoted names
        """
        self.parameters: Parameters = Parameters()
        self.initialize()
        self.parameters.update(kwargs)
        self.promoted_source_shapes: Dict[str, Shape] = dict()
        self.promoted_target_shapes: Dict[str, Shape] = dict()
        self.sources_to_promote_to_parent: Dict[str, Shape] = dict()
        self.sinks_to_promote_to_parent: Dict[str, Shape] = dict()
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
        self.promoted_to_unpromoted: Dict[str, Set[str]] = dict()
        self.unpromoted_to_promoted: Dict[str, str] = dict()

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
                f"CSDL can only print information about Variable objects. `{var}` is a `{type(var)}`."
            )
        op = print_var(var)
        out = Output(
            var.name + '_print',
            # val=var.val,
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
        if len(self.objective) > 0:
            raise ValueError(
                "Cannot add more than one objective. More than one objective was added to the same model."
            )
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
        dv_name,
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
        if dv_name not in [x.name for x in self.inputs]:
            raise KeyError(
                "{} is not an input to the main model; have you called `Model.create_input({}, **kwargs)` prior to adding this design variable?"
                .format(dv_name, dv_name))
        if dv_name in self.design_variables.keys():
            raise KeyError(
                "{} already added as a design variable".format(dv_name))
        self.design_variables[dv_name] = dict(
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
        name: str,
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
                if np.greater(lower, upper).any():
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
        if a == b:
            raise KeyError(
                f"Attempting to connect two variables named {a}. This name refers to the same variable, which cannot be connected to itself."
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
        self.declared_variable_names_set.add(v.name)
        return v

    def create_input(
        self,
        name: str,
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
        self.input_names_set.add(i.name)
        return i

    def create_output(
        self,
        name: str,
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
        return c

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
            if var in self.registered_outputs_set:
            # if var in self.registered_outputs:
                raise ValueError(
                    "Cannot register output twice; attempting to register "
                    "{} as {}.".format(var.name, name))
            # if name in [r.name for r in self.registered_outputs]:
            if name in self.registered_output_names_set:
                raise ValueError(
                    "Cannot register two outputs with the same name; attempting to register two outputs with name {}."
                    .format(name))
            if name in self.input_names_set:
                raise ValueError(
                    "Cannot register output with the same name as an input; attempting to register output named {} with same name as an input."
                    .format(name))
            if name in self.declared_variable_names_set:
                raise ValueError(
                    "Cannot register output with the same name as a declared variable; attempting to register output named {} with same name as a declared variable."
                    .format(name))

            var.name = name
            self.registered_outputs.append(var)
            self.registered_outputs_set.add(var)
            self.registered_output_names_set.add(var.name)
        if not hasattr(var, 'val'):
            var.val = np.ones(var.shape)
        return var

    def add(
        self,
        submodel: 'Model',
        name: Union[str, None] = None,
        promotes: Union[List[str], None] = None,
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
            Submodel to add to `Model`
        promotes: List
            Variables to promote

        **Returns**

        Model
            Submodel added by user
        """
        if not isinstance(submodel, Model):  # type: ignore
            raise TypeError("{} is not a Model".format(submodel))
        # if issubclass(Model, type(submodel)):
        #     raise DeprecationWarning("Adding a submodel that is an instance of the Model base class will not be allowed in future versions of CSDL. Use with self.create_submodel(\'<name>\') as <obj>` instead.")
        if name in [s.name for s in self.subgraphs]:
            raise KeyError(
                "Cannot add model with duplicate name {}".format(name))
        if type(submodel) is Model:
            warn(
                "Model named {} is not a subclass of Model. This is likely due to defining a model inline. It is recommended to define a new subclass when defining a submodel to maximiize code reuse."
                .format(name))

        subgraph = Subgraph(
            name,
            submodel,
            promotes=promotes,
        )
        self.subgraphs.append(subgraph)

        return submodel

    def _bracketed_search(
        self,
        states: Dict[str, Dict[str, Any]],
        residuals: List[str],
        implicit_model: 'Model',
        brackets: Dict[str,
                       Tuple[Union[int, float, np.ndarray, Variable],
                             Union[int, float, np.ndarray, Variable]]],
        *arguments: Variable,
        expose: List[str] = [],
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
        state_names = list(states.keys())
        (
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_residuals,
            rep,
            exposed_variables,
        ) = self._generate_maps_for_implicit_operation(
            implicit_model,
            arguments,
            state_names,
            residuals,
            expose,
        )

        # store brackets that are not CSDL variables as numpy arrays
        new_brackets: Dict[str, Tuple[Union[np.ndarray, Variable],
                                      Union[np.ndarray,
                                            Variable]]] = dict()
        # use this to check which states the user has failed to assign a
        # bracket
        states_without_brackets = copy(state_names)
        for k, v in brackets.items():
            if k not in state_names:
                raise ValueError(
                    "No state {} for specified bracket {}".format(k, v))
            if k in states_without_brackets:
                states_without_brackets.remove(k)

            if len(v) != 2:
                raise ValueError(
                    "Bracket {} for state {} is not a tuple of two values or Variable objects."
                    .format(v, k))

            (a, b) = v
            a = _to_array(a)
            b = _to_array(b)
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
            implicit_model,
            rep,
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_variables,
            exposed_residuals,
            *arguments,
            expose=expose,
            brackets=new_brackets,
            # TODO: add tol
        )

        return self._return_implicit_outputs(
            implicit_model,
            op,
            residuals,
            expose,
            states,
        )

    def _implicit_operation(
        self,
        states: Dict[str, Dict[str, Any]],
        *arguments: Variable,
        residuals: List[str],
        model: 'Model',
        nonlinear_solver: NonlinearSolver,
        linear_solver: Union[LinearSolver, None] = None,
        expose: List[str] = [],
        defaults: Dict[str, Union[int, float, np.ndarray]] = dict(),
    ) -> Union[Output, Tuple[Output, ...]]:
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
        state_names = list(states.keys())
        (
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_residuals,
            rep,
            exposed_variables,
        ) = self._generate_maps_for_implicit_operation(
            model,
            arguments,
            state_names,
            residuals,
            expose,
        )

        # store default values as numpy arrays
        new_default_values: Dict[str, np.ndarray] = dict()
        for k, v in defaults.items():
            if k not in state_names:
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
            model,
            rep,
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_variables,
            exposed_residuals,
            *arguments,
            expose=expose,
            defaults=new_default_values,
            nonlinear_solver=nonlinear_solver,
            linear_solver=linear_solver,
        )

        return self._return_implicit_outputs(
            model,
            op,
            residuals,
            expose,
            states,
        )

    def _generate_maps_for_implicit_operation(
        self,
        model: 'Model',
        arguments: Tuple[Variable, ...],
        state_names: List[str],
        residual_names: List[str],
        expose: List[str] = [],
    ) -> Tuple[Dict[str, Output], Dict[str, DeclaredVariable], Dict[
            str, List[DeclaredVariable]], Dict[
            str, List[DeclaredVariable]], Set[str],
            GraphRepresentation, Dict[str, Output]]:
        if not isinstance(model, Model):
            raise TypeError("{} is not a Model".format(model))

        rep = GraphRepresentation(model)

        # top level inputs will be unused, so we don't allow them
        if len(model.inputs) > 0:
            raise ValueError(
                "The model that defines residuals is not allowed to"
                "define top level inputs (i.e. calls to"
                "`Model.create_input`).")

        # check for duplicate arguments, states, and residuals
        arg_names: List[str] = [var.name for var in arguments]
        if len(set(arg_names)) < len(arg_names):
            raise ValueError("Duplicate arguments found")
        if len(set(state_names)) < len(state_names):
            raise ValueError("Duplicate names for states found")
        if len(set(residual_names)) < len(residual_names):
            raise ValueError("Duplicate names for residuals found")

        # check that each declared state has an associated residual
        if len(state_names) != len(residual_names):
            raise ValueError(
                "Number of states and residuals must be equal")

        for name in expose:
            if '.' in name:
                KeyError(
                    "Invalid name {} for exposing an intermediate variable in composite residual. Exposing intermediate variables with unpromoted names is not supported."
                    .format(name))

        # check that name and shape of each argument matches name and
        # shape of a declared variable in internal model, and transfer
        # value from argument to declared variable in model
        declared_variables_map: Dict[str, DeclaredVariable] = {
            x.name: x
            for x in model.declared_variables
        }
        for arg in arguments:
            if arg.name not in declared_variables_map.keys():
                raise ValueError(
                    "The argument {} is not a declared variable of the model used to define an implicit operation"
                    .format(arg.name))
            var = declared_variables_map[arg.name]
            if arg.shape != var.shape:
                raise ValueError(
                    "The argumet {} has shape {}, which does not match the shape {} of the declared variable of the model used to define an implicit operation"
                    .format(arg.name, arg.shape, var.shape))
            # arg.val = np.ones(arg.shape)
            if not hasattr(arg, 'val'):
                arg.val = np.ones(arg.shape)
            var.val = arg.val

        # check that name of each state matches name of a declared
        # variable in internal model
        for state_name in state_names:
            if state_name not in declared_variables_map.keys():
                raise ValueError(
                    "The state {} is not a declared variable of the model used to define an implicit operation"
                    .format(state_name))

        # check that name of each residual matches name of a registered
        # output in internal model
        registered_outputs_map: Dict[str, Output] = {
            x.name: x
            for x in model.registered_outputs
        }
        for residual_name in residual_names:
            if residual_name not in registered_outputs_map.keys():
                raise ValueError(
                    "The residual {} is not a registered output of the model used to define an implicit operation"
                    .format(residual_name))
        exposed_variables: Dict[str, Output] = {
            x.name: x
            for x in model.registered_outputs if x.name in set(expose)
        }

        # check that name of each exposed intermediate output matches
        # name of a registered output in internal model
        for exposed_name in expose:
            if exposed_name not in registered_outputs_map.keys():
                raise ValueError(
                    "The exposed output {} is not a registered output of the model used to define an implicit operation"
                    .format(exposed_name))

        # create two-way mapping between state objects and residual
        # objects so that back end can define derivatives of residuals
        # wrt states and arguments
        out_res_map: Dict[str, Output] = dict()
        res_out_map: Dict[str, DeclaredVariable] = dict()
        for s, r in zip(state_names, residual_names):
            if registered_outputs_map[
                    r].shape != declared_variables_map[s].shape:
                raise ValueError(
                    "Shape of state {} and residual {} do not match.".
                    format(s, r))
            out_res_map[s] = registered_outputs_map[r]
            res_out_map[r] = declared_variables_map[s]

        # TODO: (?) keep track of which exposed variables depend on
        # residuals; necessary for computing derivatives of residuals
        # associated with exposed variables wrt exposed variables, but
        # only those exposed variables that do not depend on a stata

        argument_names = [x.name for x in arguments]

        # Associate states with the arguments and states they depend on;
        out_in_map: Dict[str, List[DeclaredVariable]] = dict()
        for state_name, residual in out_res_map.items():
            # Collect inputs (terminal nodes) for this residual only; no
            # duplicates
            in_vars = list(
                set(collect_terminals(
                    set(),
                    residual,
                    residual,
                    set(),
                )))

            if state_name in state_names and state_name not in [
                    var.name for var in in_vars
            ]:
                raise ValueError(
                    "Residual {} does not depend on state {}".format(
                        residual.name, state_name))

            # Only the arguments specified in the parent model and
            # declared states will be
            # inputs to the internal model
            out_in_map[state_name] = [
                v for v in in_vars
                if v.name in set(argument_names + state_names)
            ]

        # Associate exposed outputs with the inputs they depend on;
        exp_in_map: Dict[str, List[DeclaredVariable]] = dict()
        for exposed_name in expose:
            # Collect inputs (terminal nodes) for this residual only; no
            # duplicates
            in_vars = list(
                set(
                    collect_terminals(
                        set(),
                        registered_outputs_map[exposed_name],
                        registered_outputs_map[exposed_name],
                        set(),
                    )))

            # Only the arguments specified in the parent model and
            # declared states will be
            # inputs to the internal model
            exp_in_map[exposed_name] = [
                v for v in in_vars
                if v.name in set(argument_names + state_names)
            ]

        # collect exposed variables that are residuals so that we don't
        # assume residuals are zero for these variables
        exposed_residuals: Set[str] = {
            exposed_name
            for exposed_name in expose
            if exposed_name in set(residual_names)
        }

        return (
            out_res_map,
            res_out_map,
            out_in_map,
            exp_in_map,
            exposed_residuals,
            rep,
            exposed_variables,
        )

    def _return_implicit_outputs(
        self,
        model: 'Model',
        op: Union[ImplicitOperation, BracketedSearchOperation],
        residuals: List[str],
        expose: List[str],
        states: Dict[str, Dict[str, Any]],
    ) -> Union[Output, Tuple[Output, ...]]:
        # create outputs of operation, establish dependencies on
        # operation, and register outputs
        outs: List[Output] = []

        # TODO: loop over exposed
        state_names = list(states.keys())
        for s, r in zip(state_names, residuals):

            internal_var = list(
                filter(lambda x: x.name == s,
                       model.declared_variables))[0]

            out = Output(
                s,
                shape=internal_var.shape,
                **states[s],
                op=op,
            )
            out.shape, out.val = get_shape_val(internal_var.shape, states[s]['val'])
            self.register_output(s, out)
            outs.append(out)

        # include exposed intermediate outputs in GraphRepresentation
        se = set(expose)
        ex = filter(lambda x: x.name in se, model.registered_outputs)
        for e in ex:
            if not hasattr(e, 'val'):
                e.val = np.ones(e.shape)

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
        op.outs = tuple(outs)

        # return outputs
        if len(outs) > 1:
            return tuple(outs)
        else:
            return outs[0]

    def create_implicit_operation(self, model: 'Model'):
        return ImplicitOperationFactory(self, model)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.define()

    @contextmanager
    def create_submodel(self,
                        name: str,
                        promotes: Union[List[str], None] = None):
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
            with InlineModel() as m:
                yield m
        finally:
            self.add(m, name=name, promotes=promotes)


class InlineModel(Model):
    """
    Subclass of Model to use within create_submodel context to avoid
    creating instances of Model base class
    """
    pass
