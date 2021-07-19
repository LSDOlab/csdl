from collections.abc import Iterable
from contextlib import contextmanager
from typing import Callable, Dict, Tuple, List, Union, Set

# from csdl.core._group import _Group
from csdl.core.explicit_output import ExplicitOutput
from csdl.utils.graph import (
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
from csdl.utils.set_default_values import set_default_values
from warnings import warn
import networkx as nx
import matplotlib.pyplot as plt


def register_nodes(nodes, node):
    for n in node.dependencies:
        nodes[n._id] = n
        register_nodes(nodes, n)


def build_clean_dag(registered_outputs):
    nodes = dict()
    for r in registered_outputs:
        nodes[r._id] = r
    for r in registered_outputs:
        register_nodes(nodes, r)

    # Clean up graph, removing dependencies that do not constrain
    # execution order
    for node in nodes.values():
        remove_indirect_dependencies(node)
        if isinstance(node, Operation):
            if len(node.dependencies) < 1:
                raise ValueError(
                    "Operation objects must have at least one dependency")


# TODO: collect all inputs from all models to ensure that the entire
# model does have inputs; issue warning if not
def _build_intermediate_representation(func: Callable) -> Callable:
    """
    This function replaces ``Group.setup`` with a new method that calls
    ``Group.setup`` and performs the necessary steps to determine
    execution order and construct and add the appropriate subsystems.

    The new method is the core of the ``csdl`` package. This function
    analyzes the Directed Acyclic Graph (DAG) and sorts expressions.
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

            # add forward edges
            for r in self.registered_outputs:
                r.add_fwd_edges()

            # remove_duplicate_nodes(self.nodes, self.registered_outputs)

            # combine elementwise operations and use complex step
            terminate = False
            if len(self.registered_outputs) == 0:
                terminate = True
            while terminate is False:
                for r in self.registered_outputs:
                    terminate = combine_operations(self.registered_outputs, r)
                terminate = True

            # Create record of all nodes in DAG
            for r in self.registered_outputs:
                self.nodes[r._id] = r
            for r in self.registered_outputs:
                register_nodes(self.nodes, r)

            if True:
                # Use modified Kahn's algorithm to sort nodes, reordering
                # expressions except where user registers outputs out of order
                self.sorted_expressions = modified_topological_sort(
                    self.registered_outputs)
            # else:
            # Use Kahn's algorithm to sort nodes, reordering
            # expressions without regard for the order in which user
            # registers outputs
            # self.sorted_expressions =
            # topological_sort(self.registered_outputs)

            # Check that all outputs are defined
            for output in self.sorted_expressions:
                if isinstance(output, (ExplicitOutput)):
                    if output.defined is False:
                        raise ValueError("Output not defined for {}".format(
                            repr(output)))

            # Define child models recursively
            for subgraph in self.subgraphs:
                subgraph.submodel.define()

            _, _ = set_default_values(self)

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
        self.sorted_expressions = []
        self.reverse_branch_sorting: bool = False
        self._most_recently_added_subgraph: Subgraph = None
        self.brackets_map = None
        self.out_vals = dict()
        self.subgraphs: List[Subgraph] = []
        self.variables_promoted_from_children: List[Variable] = []
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
        self.parameters.update(kwargs)
        self.linear_solver = None
        self.nonlinear_solver = None

    def initialize(self):
        """
        User defined method to set parameters
        """
        pass

    def define(self):
        """
        User defined method to define numerical model
        """
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
            Name of variable in CSDL to be used as a local input that
            takes a value from a parent model, child model, or
            previously registered output within the model.
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
        Create an input to the main model, whose value remains constant
        during model evaluation.

        Parameters
        ----------
        name: str
            Name of variable in CSDL
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
        Create a value that is computed explicitly, either through
        indexed assignment, or as a fixed point iteration.

        Parameters
        ----------
        name: str
            Name of variable in CSDL
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
            Name of variable in CSDL

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
        from csdl.core.implicit_model import ImplicitModel
        if not isinstance(submodel, (Model, ImplicitModel, CustomOperation)):
            raise TypeError(
                "{} is not a Model, ImplicitModel, or CustomOperation".format(
                    submodel))

        if name == '':
            name = gen_hex_name(Model._count),

        subgraph = Subgraph(
            name,
            submodel,
            promotes=promotes,
            promotes_inputs=promotes_inputs,
            promotes_outputs=promotes_outputs,
        )
        self.subgraphs.append(subgraph)
        if self._most_recently_added_subgraph is not None:
            subgraph.add_dependency_node(self._most_recently_added_subgraph)
        self._most_recently_added_subgraph = subgraph
        for r in self.registered_outputs:
            if not isinstance(r, Subgraph):
                self._most_recently_added_subgraph.add_dependency_node(r)
        for i in self.inputs:
            self._most_recently_added_subgraph.add_dependency_node(i)

        # Add subystem to DAG
        self.registered_outputs.append(subgraph)

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

    def visualize_sparsity(self):
        import matplotlib.pylab as plt
        import scipy.sparse as sparse

        self.define()

        # initialize sparse matrix
        n = len(self.sorted_expressions)
        A = sparse.lil_matrix((n, n))

        # populate diagonals to indicate nodes
        indices = dict()
        for i, node in enumerate(reversed(self.sorted_expressions)):
            print(node.name)
            indices[node] = i
            A[i, i] = 1

        # populate diagonals to indicate connections
        for i, node in enumerate(reversed(self.sorted_expressions)):
            for dep in node.dependencies:
                if dep in indices.keys():
                    A[indices[dep], i] = 1

        plt.spy(A.tocoo())
        plt.show()

    def visualize_graph(self):

        G = nx.DiGraph()

        names = [node.name for node in self.nodes.values()]

        G.add_nodes_from(names)
        G.add_node('BEGIN')
        G.add_node('END')

        edge_tuples = []
        for r in self.registered_outputs:
            edge_tuples.append((r.name, 'END', 1))
        for i in self.inputs:
            edge_tuples.append(('BEGIN', i.name, 1))
        for node in self.nodes.values():
            for dep in node.dependencies:
                edge_tuples.append((dep.name, node.name, 1))
        G.add_weighted_edges_from(edge_tuples)

        variables = [
            node for node in list(
                filter(lambda x: isinstance(x, Variable), self.nodes.values()))
        ]
        operations = [
            node for node in list(
                filter(lambda x: isinstance(x, (Operation, Subgraph)),
                       self.nodes.values()))
        ]

        pos = dict()
        pos.update((n.name, (1, i)) for i, n in enumerate(variables))
        pos.update((n.name, (2, i + 1)) for i, n in enumerate(operations))
        pos['BEGIN'] = (2, 0)
        pos['END'] = (2, len(operations) + 1)
        nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
        plt.show()
