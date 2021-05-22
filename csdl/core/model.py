from collections.abc import Iterable
from contextlib import contextmanager
from typing import Callable, Dict, Tuple, List, Union, Set

# from csdl.core._group import _Group
from csdl.core.explicit_output import ExplicitOutput
from csdl.core.graph import (
    remove_indirect_dependencies,
    modified_topological_sort,
    # remove_duplicate_nodes,
)
from csdl.core.implicit_output import ImplicitOutput
from csdl.core.node import Node
from csdl.core.variable import Variable
from csdl.core.input import Input
from csdl.core.output import Output
from csdl.core.operation import Operation
from csdl.core.custom_operation import CustomOperation
from csdl.core.subgraph import Subgraph
from csdl.operations.print_var import print_var
from csdl.utils.gen_hex_name import gen_hex_name
from csdl.utils.parameters import Parameters
from csdl.utils.combine_operations import combine_operations
from warnings import warn


# TODO: collect all inputs from all models to ensure that the entire
# model does have inputs; issue warning if not
def _build_intermediate_representation(func: Callable) -> Callable:
    """
    This function replaces ``Group.setup`` with a new method that calls
    ``Group.setup`` and performs the necessary steps to determine
    execution order and construct and add the appropriate subsystems.

    The new method is the core of the ``csdl`` package. This function
    analyzes the Directed Acyclic Graph (DAG), sorts expressions, and
    directs OpenMDAO to add the corresponding ``Component`` objects.

    This function ensures an execution order that is free of unnecessary
    feedback regardless of the order in which the user registers
    outputs.
    """
    def _sort_nodes(self):
        """
        User defined method to define expressions and add subsystems for
        model execution
        """
        if self._defined is False:
            self._defined = True
            func(self)

            # Check if all design variables are inputs
            input_names = set(
                filter(lambda x: isinstance(x, Input),
                       [inp.name for inp in self.inputs]))
            for name in self.design_variables:
                if name not in input_names:
                    raise KeyError(
                        "{} is not an input to the model".format(name))
            del input_names

            # check that all connections are valid
            # output_names = set([n.name for n in nodes if isinstance(n, Output)], )
            # output_shapes = set([(n.name, n.shape)
            #                      for n in nodes if isinstance(n, Output)], )
            # variable_names = set([
            #     n.name for n in nodes
            #     if isinstance(n, Variable) and not isinstance(n, Output)
            # ], )
            # variable_shapes = set(
            #     [(n.name, n.shape) for n in nodes
            #      if isinstance(n, Variable) and not isinstance(n, Output)], )
            # for (a, b) in self.connections:
            #     if a not in output_names:
            #         raise KeyError(
            #             "No output named {} for connection from {} to {}".format(
            #                 a, a, b))
            #     if b not in input_names:
            #         raise KeyError(
            #             "No input named {} for connection from {} to {}".format(
            #                 b, a, b))

            # Create a record of all nodes in DAG
            for r in self.registered_outputs:
                r.register_nodes(self.nodes)

            # Clean up graph, removing dependencies that do not constrain
            # execution order
            for node in self.nodes.values():
                remove_indirect_dependencies(node)
                if isinstance(node, Operation):
                    if len(node.dependencies) < 1:
                        raise ValueError(
                            "Operation objects must have at least one dependency"
                        )

            # add forward edges
            for r in self.registered_outputs:
                r.add_fwd_edges()

            # remove_duplicate_nodes(self.nodes, self.registered_outputs)

            # combine elementwise operations and use complex step
            repeat = True
            while repeat is True:
                for r in self.registered_outputs:
                    repeat = combine_operations(self.registered_outputs, r)

            # remove unused expressions
            keys = []
            for name, node in self.nodes.items():
                if len(node.dependents) == 0:
                    keys.append(name)
            for name in keys:
                del self.nodes[name]

            if True:
                # Use modified Kahn's algorithm to sort nodes, reordering
                # expressions except where user registers outputs out of order
                self.sorted_expressions = modified_topological_sort(
                    self.registered_outputs)
            # else:
            # Use Kahn's algorithm to sort nodes, reordering
            # expressions without regard for the order in which user
            # registers outputs
            # self.sorted_expressions = topological_sort(self.registered_outputs)

            # Check that all outputs are defined
            for output in self.sorted_expressions:
                if isinstance(output, (ExplicitOutput)):
                    if output.defined is False:
                        raise ValueError("Output not defined for {}".format(
                            repr(output)))

            # TODO: front end defines models recursively, not back end
            # Define child models recursively
            for submodel in self.submodels:
                submodel.define()

    return _sort_nodes


class _ComponentBuilder(type):
    def __new__(cls, name, bases, attr):
        attr['define'] = _build_intermediate_representation(attr['define'])
        return super(_ComponentBuilder, cls).__new__(cls, name, bases, attr)


class Model(metaclass=_ComponentBuilder):
    _count = -1

    def __init__(self, **kwargs):
        Model._count += 1
        self._defined = False
        self.nodes: dict = {}
        self.input_vals: dict = {}
        self.sorted_builders = []
        self.reverse_branch_sorting: bool = False
        self._most_recently_added_subgraph: Subgraph = None
        self.brackets_map = None
        self.out_vals = dict()
        self.inputs: List[Input] = []
        self.variables: List[Variable] = []
        self.registered_outputs: List[Union[Output, ExplicitOutput,
                                            Subgraph]] = []
        self.objective = None
        self.constraints = dict()
        self.design_variables = dict()
        self.connections: List[Tuple[str, str]] = []
        self.parameters = Parameters()
        self.initialize()
        self.submodels = []
        self.linear_solver = None
        self.nonlinear_solver = None

    def initialize(self):
        """
        User defined method to set options
        """
        pass

    def define(self):
        pass

    def print_var(self, var: Variable):
        if not isinstance(var, Variable):
            raise TypeError(
                "CSDL can only print information about Variable objects")
        op = print_var(var)
        out = Output(
            var.name + '_print',
            val=var.val,
            shape=var.shape,
            src_indices=var.src_indices,
            flat_src_indices=var.flat_src_indices,
            units=var.units,
            desc=var.desc,
            tags=var.tags,
            shape_by_conn=var.shape_by_conn,
            copy_shape=var.copy_shape,
            distributed=var.distributed,
            op=op,
        )
        out.add_dependency_node(op)
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
        vectorize_derivs=False,
        cache_linear_solution=False,
    ):
        self.objective = dict(
            name=name,
            ref=ref,
            ref0=ref0,
            index=index,
            units=units,
            adder=adder,
            scaler=scaler,
            parallel_deriv_color=parallel_deriv_color,
            vectorize_derivs=vectorize_derivs,
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
        vectorize_derivs=False,
        cache_linear_solution=False,
    ):
        if name in self.design_variables.keys():
            raise ValueError(
                "{} already added as a design variable".format(name))
        else:
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
                vectorize_derivs=vectorize_derivs,
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
        vectorize_derivs=False,
        cache_linear_solution=False,
    ):
        if name in self.constraints.keys():
            raise ValueError("Constraint already defined for {}".format(name))
        else:
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
                vectorize_derivs=vectorize_derivs,
                cache_linear_solution=cache_linear_solution,
            )

    def connect(self, a: str, b: str):
        warn("Error messages for connections are not yet built into "
             "CSDL frontend. Pay attention to any errors emmited by backend.")
        if (a, b) in self.connections:
            warn("Connection from {} to {} issued twice.".format(a, b))
        self.connections.append((a, b))

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
    ) -> Variable:
        """
        Declare an input to use in an expression.

        An input can be an output of a child ``System``. If the user
        declares an input that is computed by a child ``System``, then
        the call to ``self.declare_variable`` must appear after the call to
        ``self.add``.

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO to be used as an input in
            generated ``Component`` objects
        shape: Tuple[int]
            Shape of variable
        val: Number or ndarray
            Default value for variable

        Returns
        -------
        DocInput
            An object to use in expressions
        """
        v = Variable(
            name,
            val=val,
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
        if self._most_recently_added_subgraph is not None:
            v.add_dependency_node(self._most_recently_added_subgraph)
        self.variables.append(v)
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
        Create a value that is constant during model evaluation

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO to be computed by
            ``ExplicitComponent`` objects connected in a cycle, or by an
            ``ExplicitComponent`` that concatenates variables
        shape: Tuple[int]
            Shape of variable
        val: Number or ndarray
            Value for variable during first model evaluation
        dv: bool
            Flag to set design variable

        Returns
        -------
        Input
            An object to use in expressions
        """
        i = Input(
            name,
            val=val,
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

    # TODO: add solver argument to create a model from cyclic
    # relationships
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
    ) -> ExplicitOutput:
        """
        Create a value that is computed explicitly

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO to be computed by
            ``ExplicitComponent`` objects connected in a cycle, or by an
            ``ExplicitComponent`` that concatenates variables
        shape: Tuple[int]
            Shape of variable

        Returns
        -------
        ExplicitOutput
            An object to use in expressions
        """
        ex = ExplicitOutput(
            name,
            val=val,
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
        if ex in self.registered_outputs:
            raise ValueError("cannot create the same output in the same model")
        self.registered_outputs.append(ex)
        return ex

    def register_output(self, name: str, var: Variable) -> Variable:
        """
        Register ``expr`` as an output of the ``Group``.
        When adding subsystems, each of the subsystem's inputs requires
        a call to ``register_output`` prior to the call to
        ``add``.

        Parameters
        ----------
        name: str
            Name of variable in OpenMDAO

        expr: Variable
            Variable that computes output

        Returns
        -------
        Variable
            Variable that computes output
        """
        if not isinstance(var, Variable):
            raise TypeError(
                "Attempting to register Variable {} as an output."
                "Can only register Variable object as an output".format(
                    var.name))
        if not isinstance(var, Output):
            warn(
                "Registering variable {} has no effect, as it does not depend on an operation."
                .format(var.name))
        else:
            if var in self.registered_outputs:
                raise ValueError(
                    "Cannot register output twice; attempting to register "
                    "{} as {}.".format(var.name, name))

            var.name = name
            self.registered_outputs.append(var)
        return var

    def add(
        self,
        submodel,
        name: str = '',
        promotes: Iterable = None,
        promotes_inputs: Iterable = None,
        promotes_outputs: Iterable = None,
    ):
        """
        Add a subsystem to the ``Group``.

        ``self.add`` call must be preceded by a call to
        ``self.register_output`` for each of the subsystem's inputs,
        and followed by ``self.declare_variable`` for each of the
        subsystem's outputs.

        Parameters
        ----------
        name: str
            Name of subsystem
        submodel: System
            Subsystem to add to `Group`
        promotes: Iterable
            Variables to promote
        promotes_inputs: Iterable
            Inputs to promote
        promotes_outputs: Iterable
            Outputs to promote

        Returns
        -------
        System
            Subsystem to add to `Group`
        """
        if not isinstance(submodel, (Model, CustomOperation)):
            raise TypeError(
                "{} is not a Model or a CustomOperation".format(submodel))

        if name == '':
            name = gen_hex_name(Model._count),

        self._most_recently_added_subgraph = Subgraph(
            name,
            submodel,
            promotes=promotes,
            promotes_inputs=promotes_inputs,
            promotes_outputs=promotes_outputs,
        )
        for r in self.registered_outputs:
            self._most_recently_added_subgraph.add_dependency_node(r)

        # Add subystem to DAG
        self.registered_outputs.append(self._most_recently_added_subgraph)

        self.submodels.append(submodel)
        return submodel

    @contextmanager
    def create_model(self, name: str):
        """
        Create a ``Group`` object and add as a subsystem, promoting all
        inputs and outputs.
        For use in ``with`` contexts.
        NOTE: Only use if planning to promote all varaibales within
        child ``Group`` object.

        Parameters
        ----------
        name: str
            Name of new child ``Group`` object

        Returns
        -------
        Group
            Child ``Group`` object whosevariables are all promoted
        """
        try:
            m = Model()
            self.add(m, name=name, promotes=['*'])
            yield m
        finally:
            # m.define()
            pass
