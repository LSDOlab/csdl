from collections.abc import Iterable
from contextlib import contextmanager
from typing import Callable, Tuple, List, Union

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
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sparse


def build_symbol_table(symbol_table, node):
    for n in node.dependencies:
        symbol_table[n._id] = n
        build_symbol_table(symbol_table, n)


def build_clean_dag(registered_outputs):
    symbol_table = dict()
    for r in registered_outputs:
        symbol_table[r._id] = r
    for r in registered_outputs:
        build_symbol_table(symbol_table, r)

    # Clean up graph, removing dependencies that do not constrain
    # execution order
    for node in symbol_table.values():
        remove_indirect_dependencies(node)
        if isinstance(node, Operation):
            if len(node.dependencies) < 1:
                raise ValueError(
                    "Operation objects must have at least one dependency")


# TODO: collect all inputs from all models to ensure that the entire
# model does have inputs; issue warning if not
def _run_front_end_and_middle_end(run_front_end: Callable) -> Callable:
    """
    This function replaces ``Group.setup`` with a new method that calls
    ``Group.setup`` and performs the necessary steps to determine
    execution order and construct and add the appropriate subsystems.

    The new method is the core of the ``csdl`` package. This function
    analyzes the Directed Acyclic Graph (DAG) and sorts expressions.
    """
    def _run_middle_end(self):
        """
        User defined method to define expressions and add subsystems for
        model execution
        """
        if self._defined is False:
            self._defined = True
            run_front_end(self)

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

            # Check that all outputs are defined
            # for output in self.sorted_expressions:
            for output in self.registered_outputs:
                if isinstance(output, ExplicitOutput):
                    if output.defined is False:
                        raise ValueError("Output not defined for {}".format(
                            repr(output)))

            # add forward edges; nodes with fewer forward edges than
            # dependents will be ignored when sorting nodes
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
                self.symbol_table[r._id] = r
            for r in self.registered_outputs:
                build_symbol_table(self.symbol_table, r)

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

            # Define child models recursively
            for subgraph in self.subgraphs:
                if not isinstance(subgraph.submodel, CustomOperation):
                    subgraph.submodel.define()

            _, _ = set_default_values(self)

    return _run_middle_end


class _CompilerFrontEndMiddleEnd(type):
    def __new__(cls, name, bases, attr):
        attr['define'] = _run_front_end_and_middle_end(attr['define'])
        return super(_CompilerFrontEndMiddleEnd,
                     cls).__new__(cls, name, bases, attr)


class Model(metaclass=_CompilerFrontEndMiddleEnd):
    _count = -1

    def __init__(self, **kwargs):
        Model._count += 1
        self._defined = False
        self.symbol_table: dict = {}
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
        var,
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
        name = var.name
        if not isinstance(var, Output):
            raise TypeError('Variable must be an Output')
        if not var.shape == (1, ):
            raise ValueError('Variable must be a scalar')
        if var._id == var.name:
            raise NameError(
                'Variable is not named by user. Name Variable by registering it as an output first.'
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
        var,
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
        variable must be an ``Input``. This will signal to the optimizer
        that it is responsible for updating the input variable.
        """
        name = var.name
        if not isinstance(var, Input):
            raise TypeError('Variable must be an Input')
        if isinstance(var, Output):
            raise TypeError('Variable is not an input to the model')
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
                cache_linear_solution=cache_linear_solution,
            )

    def add_constraint(
        self,
        var,
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
        name = var.name
        if name in self.constraints.keys():
            raise ValueError("Constraint already defined for {}".format(name))
        else:
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

        # promote by default
        if promotes == [] and (promotes_inputs
                               is not None) or (promotes_outputs is not None):
            raise ValueError(
                "cannot selectively promote inputs and outputs if promotes=[]")
        if promotes == ['*'] and (promotes_inputs is not None) or (
                promotes_outputs is not None):
            raise ValueError(
                "cannot selectively promote inputs and outputs if promoting all variables"
            )
        if promotes is None and (promotes_inputs is None) and (promotes_outputs
                                                               is None):
            promotes = ['*']
        if promotes == []:
            promotes_inputs = []

        subgraph = Subgraph(
            gen_hex_name(Node._count + 1) if name == '' else name,
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
        """
        Visualize the sparsity pattern of jacobian for this model
        """
        self.define()
        nodes = self.sorted_expressions
        n = len(nodes)
        A = sparse.lil_matrix((n, n))
        A, _, indices, implicit_nodes = add_diag(A, nodes)
        A = add_off_diag(A, self, indices)
        A = add_off_diag_implicit(A, indices, implicit_nodes)
        plt.spy(A, markersize=1)
        ax = plt.gca()
        for axis in 'left', 'right', 'bottom', 'top':
            ax.spines[axis].set_linewidth(2)

        plt.show()

    def visualize_graph(self):
        import networkx as nx
        G = nx.DiGraph()

        names = [node.name for node in self.symbol_table.values()]

        G.add_nodes_from(names)
        G.add_node('BEGIN')
        G.add_node('END')

        edge_tuples = []
        for r in self.registered_outputs:
            edge_tuples.append((r.name, 'END', 1))
        for i in self.inputs:
            edge_tuples.append(('BEGIN', i.name, 1))
        for node in self.symbol_table.values():
            for dep in node.dependencies:
                edge_tuples.append((dep.name, node.name, 1))
        G.add_weighted_edges_from(edge_tuples)

        variables = [
            node for node in list(
                filter(lambda x: isinstance(x, Variable),
                       self.symbol_table.values()))
        ]
        operations = [
            node for node in list(
                filter(lambda x: isinstance(x, (Operation, Subgraph)),
                       self.symbol_table.values()))
        ]

        pos = dict()
        pos.update((n.name, (1, i)) for i, n in enumerate(variables))
        pos.update((n.name, (2, i + 1)) for i, n in enumerate(operations))
        pos['BEGIN'] = (2, 0)
        pos['END'] = (2, len(operations) + 1)
        nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
        plt.show()


def add_diag_implicit(A,
                      variables,
                      implicit_outputs,
                      indices=dict(),
                      implicit_nodes=dict(),
                      p=0,
                      indent=''):
    i = p
    implicit_operator = Node()
    implicit_nodes[implicit_operator] = dict()
    implicit_nodes[implicit_operator]['vars'] = variables
    implicit_nodes[implicit_operator]['outs'] = implicit_outputs
    for node in variables:
        print(indent + str(i), node.name)
        A[i, i] = 1
        # NOTE: the keys are nodes, not names of nodes
        # NOTE: there will be multiple keys with same .name
        # attribute
        indices[node] = i
        i += 1
    A[i, i] = 1
    indices[implicit_operator] = i
    i += 1
    for node in implicit_outputs:
        print(indent + str(i), node.name)
        A[i, i] = 1
        # NOTE: the keys are nodes, not names of nodes
        # NOTE: there will be multiple keys with same .name
        # attribute
        indices[node] = i
        i += 1
    return A, i, indices, implicit_nodes


def add_off_diag_implicit(A, indices, implicit_nodes):
    print(implicit_nodes)
    for implicit_operator in implicit_nodes.keys():
        print(implicit_operator)
        for node in implicit_nodes[implicit_operator]['vars']:
            print(node.name)
            print(indices[implicit_operator])
            print(indices[node])
            A[indices[node], indices[implicit_operator]] = 1
        for node in implicit_nodes[implicit_operator]['outs']:
            A[indices[implicit_operator], indices[node]] = 1
    return A


def add_diag(A, nodes, indices=dict(), implicit_nodes=dict(), p=0, indent=''):
    from csdl.core.implicit_model import ImplicitModel
    # NOTE: A must have shape (len(nodes), len(nodes))
    print(p)
    i = p
    for node in reversed(nodes):
        if i < A.get_shape()[0]:
            print(indent + str(i), node.name)
            if isinstance(node, Subgraph):
                # TODO: check CustomOperation
                if isinstance(node.submodel, Model):
                    m = len(node.submodel.sorted_expressions)
                    q = A.get_shape()[0] + m - 1
                    C = sparse.lil_matrix((q, q))
                    C[:i, :i] = A[:i, :i]
                    print(indent + str(i), node.name, A.get_shape(),
                          C.get_shape())
                    A, i, indices, implicit_nodes = add_diag(
                        C,
                        node.submodel.sorted_expressions,
                        implicit_nodes=implicit_nodes,
                        p=i,
                        indent=indent + ' ',
                    )
                elif isinstance(node.submodel, ImplicitModel):
                    variables = list(
                        filter(lambda x: len(x.dependencies) == 0,
                               node.submodel._model.variables))
                    m = len(variables) + len(
                        node.submodel.implicit_outputs) + 1
                    if m > 0:
                        q = A.get_shape()[0] + m - 1
                        C = sparse.lil_matrix((q, q))
                        print((q, q), A.get_shape(), C.get_shape())
                        C[:i, :i] = A[:i, :i]
                        A, i, indices, implicit_nodes = add_diag_implicit(
                            C,
                            variables,
                            node.submodel.implicit_outputs,
                            indices=indices,
                            implicit_nodes=implicit_nodes,
                            p=i,
                            indent=indent + ' ',
                        )
            else:
                print(indent + str(i), node.name, A.get_shape())
                A[i, i] = 1
                # NOTE: the keys are nodes, not names of nodes
                # NOTE: there will be multiple keys with same .name
                # attribute
                indices[node] = i
                i += 1
    return A, i, indices, implicit_nodes


# TODO: make triangular only if there are no cycles
def add_off_diag(A, model, indices):
    for node in reversed(model.sorted_expressions):
        if isinstance(node, Subgraph):
            if isinstance(node.submodel, Model):
                add_off_diag(A, node.submodel, indices)
        else:
            for dep in node.dependencies:
                if node in indices.keys() and dep in indices.keys():
                    if dep.name == node.name:
                        A[indices[node], indices[dep]] = 1
                    else:
                        A[indices[dep], indices[node]] = 1
    return A
