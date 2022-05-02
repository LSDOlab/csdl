try:
    from csdl.lang.model import Model
except ImportError:
    pass
from networkx import DiGraph, adjacency_matrix, dag_longest_path_length
from typing import List, Dict, Literal, Tuple, Any
from csdl.lang.input import Input
from csdl.lang.output import Output
from csdl.lang.operation import Operation
from csdl.lang.variable import Variable
from csdl.lang.custom_explicit_operation import CustomExplicitOperation
from csdl.lang.custom_implicit_operation import CustomImplicitOperation
import matplotlib.pyplot as plt
from networkx import DiGraph
from csdl.rep.ir_node import IRNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.construct_unflat_graph import construct_unflat_graph
from csdl.rep.construct_flat_graph import construct_flat_graph
from csdl.rep.variable_node import VariableNode
from typing import List, Tuple, Dict, Set, Iterable
from csdl.lang.concatenation import Concatenation
from csdl.rep.collect_promoted_variable_names import collect_promoted_variable_names
from csdl.rep.check_for_promotion_induced_cycles import check_for_promotion_induced_cycles
from csdl.rep.issue_user_specified_connections import issue_user_specified_connections
from csdl.rep.issue_user_specified_connections import issue_user_specified_connections
from csdl.rep.collect_connections import collect_connections
from csdl.rep.detect_cycles_in_connections import detect_cycles_in_connections
from csdl.rep.add_model_var_dependencies import add_model_var_dependencies_due_to_promotions, add_model_var_dependencies_due_to_connections
from csdl.rep.ir_node import IRNode
from csdl.rep.sort_nodes_nx import sort_nodes_nx
from csdl.rep.get_registered_outputs_from_graph import get_registered_outputs_from_graph
from networkx import DiGraph, ancestors
try:
    from csdl.rep.apply_fn_to_implicit_operation_nodes import apply_fn_to_implicit_operation_nodes
except ImportError:
    pass
from csdl.lang.define_recursively import define_recursively
import numpy as np


def nargs(
    graph: DiGraph,
    op: OperationNode,
    vectorized: bool,
) -> int:
    if vectorized is False:
        return np.sum(_var_sizes(graph.predecessors(op)))
    else:
        return graph.in_degree(op)


def nouts(
    graph: DiGraph,
    op: OperationNode,
    vectorized: bool,
) -> int:
    if vectorized is False:
        return np.sum(_var_sizes(graph.successors(op)))
    else:
        return graph.out_degree(op)


def _var_sizes(variable_nodes: list[VariableNode], ) -> list[int]:
    """
    Compute the number of scalar values in each vectorized variable in
    `variable_nodes`
    """
    var_sizes = [np.prod(x.var.shape) for x in variable_nodes]
    return var_sizes


def _visualize(rep: 'GraphRepresentation'):
    """
    Visualize the flattened graph for the main model containing
    variable/operation nodes and edges
    """
    plt.spy(adjacency_matrix(rep.flat_graph), markersize=1)
    plt.show()


class GraphRepresentation:
    """
    The intermediate representation of a CSDL Model, stored as a
    directed acyclic graph. The intermediate representation contains
    nodes representing variables and operations. The subgraph node is
    also used to condense graphs representing submodels to encode
    hierarchy. An intermediate representation may also be flattened to
    encode hierarchy without the use of subgraph nodes.
    """

    def __init__(self, model: 'Model'):
        if model.defined is False:
            _ = define_recursively(model)

        # PROMOTIONS
        _, _, _ = collect_promoted_variable_names(model)
        check_for_promotion_induced_cycles(model)
        add_model_var_dependencies_due_to_promotions(model)

        # CONNECTIONS
        for promoted_name, unpromoted_set in model.promoted_to_unpromoted:
            for unpromoted_name in unpromoted_set:
                model.unpromoted_to_promoted[
                    unpromoted_name] = promoted_name
        issue_user_specified_connections(model)
        self.connections: list[Tuple[str,
                                     str]] = collect_connections(model)
        """
        Connections issued to models across model hierarchy branches,
        for use by back ends that use `unflat_graph` only
        """
        detect_cycles_in_connections(self.connections)
        # TODO: implement add_model_var_dependencies_due_to_connections
        # check for cycles the same way as in add_model_var_dependencies_due_to_promotions
        add_model_var_dependencies_due_to_connections(model)

        self.unflat_graph: DiGraph = construct_unflat_graph(
            model.inputs,
            model.registered_outputs,
            model.subgraphs,
        )
        """
        Directed acyclic graph representing model.
        Each model in the model hierarchy will contain an instance of
        `IntermediateRepresentation` with `unflat_graph: DiGraph`.
        """
        self.unflat_sorted_nodes: list[IRNode] = sort_nodes_nx(
            self.unflat_graph,
            flat=False,
        )
        """
        Nodes sorted in order of execution, using the unflattened graph
        """
        self.flat_graph: DiGraph = construct_flat_graph(
            model,
            self.unflat_graph,
        )
        """
        Flattened directed acyclic graph representing main model.
        Only the main model will contain an instance of
        `IntermediateRepresentation` with `flat_graph: DiGraph`.
        All submodels in the hierarchy will contain `flat_graph = None`.
        """
        self.flat_sorted_nodes: list[IRNode] = sort_nodes_nx(
            self.flat_graph,
            flat=True,
        )
        """
        Nodes sorted in order of execution, using the flattened graph
        """
        # TODO: collect_design_variables
        self.design_variables: Dict[str, Dict[str, Any]] = dict()
        """
        Design variables of the optimization problem, if an optimization
        problem is defined
        """
        # TODO: find_objective
        self.objective: Dict[str, Any] | None = None
        """
        Objective of the optimization problem, if an optimization
        problem is defined
        """
        # TODO: collect_constraints
        self.constraints: Dict[str, Dict[str, Any]] = dict()
        """
        Constraints of the optimization problem, if a constrained
        optimization problem is defined
        """

    def input_nodes(self) -> List[VariableNode]:
        """
        Return nodes that represent inputs to the main model.
        """
        variable_nodes = self._variable_nodes()
        input_nodes: Iterable[VariableNode] = filter(
            lambda x: isinstance(x.var, Input), variable_nodes)
        return list(input_nodes)

    def output_nodes(self) -> List[VariableNode]:
        """
        Return nodes that represent outputs of the main model.
        """
        variable_nodes = self._variable_nodes()
        output_nodes: Iterable[VariableNode] = filter(
            lambda x: isinstance(x.var, Output), variable_nodes)
        return list(output_nodes)

    def operation_nodes(
        self,
        ignore_custom: bool = False,
    ) -> List[OperationNode]:
        """
        Return nodes that represent operations within the model. Uses
        flattened representation to gather operations.
        """
        return list(self._operation_nodes(ignore_custom))

    def variable_nodes(self) -> List[VariableNode]:
        """
        Return nodes that represent all variables within the model. Uses
        flattened representation to gather variables.
        """
        return list(self._variable_nodes())

    def num_inputs(self) -> int:
        """
        Total number of inputs; equivalent to `len(IntermediateRepresentation.input_nodes())`.
        """
        raise NotImplementedError

    def num_outputs(self, vectorized: bool = False) -> int:
        """
        Total number of outputs; equivalent to `len(IntermediateRepresentation.output_nodes())`.
        """
        if vectorized is True:
            return len(self.output_nodes())
        else:
            return np.sum(_var_sizes(self.output_nodes()))

    def num_operation_nodes(
        self,
        include: Dict[str, bool] | None = None,
        all: bool = True,
    ) -> int:
        """
        include
        : properties that operations must have to be included

        all
        : whether each operation must have all properties (True) specified in
        `include` or at least one property (False); default is True

        Properties to specify:
        - nonlinear
        - linear
        """
        return len(self.operation_nodes())

    def num_variable_nodes(self) -> int:
        """
        Number of variable nodes; each node represents an n-dimensional
        array specified by user. Equivalent to
        `len(IntermediateRepresentation.variable_nodes())`.
        """
        return len(self.variable_nodes())

    def predict_memory_footprint(
        self,
        mode: Literal['fwd', 'rev'],
    ) -> int:
        """
        Total number of scalar values in model, including variables from
        ImplicitOperation nodes, and residuals required for solving
        implicit operations.
        """
        if mode == 'rev':
            return np.sum(_var_sizes(list(self._variable_nodes())))
        elif mode == 'fwd':
            # need upper triangular adjacency matrix
            return 0

    def predict_computation_time(self, parallel: bool = False) -> int:
        """
        Predict computation time to evaluate the model. If parallel is
        false, then computation time is time to evaluate all operations
        in series.  If parallel is True, then computation time for a
        fully parallelized model is equivalent to computation time for
        all operations on the critical path of the graph.
        """
        if parallel is True:
            return dag_longest_path_length(self.flat_graph)
        else:
            return self.num_variable_nodes()

    def avg_arguments_per_operation(
        self,
        ignore_custom: bool = False,
        vectorized: bool = False,
    ) -> float:
        """
        Compute average number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the average.
        """
        nargs_per_op = [
            nargs(self.flat_graph, op, vectorized) for op in ops
        ]
        return float(np.sum(nargs_per_op)) / float(len(nargs_per_op))

    def avg_outputs_per_operation(
        self,
        ignore_custom: bool = False,
        vectorized: bool = False,
    ) -> float:
        """
        Compute average number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the average.
        """
        ops = self._operation_nodes(ignore_custom)
        nouts_per_op = [
            nouts(self.flat_graph, op, vectorized) for op in ops
        ]
        return float(np.sum(nouts_per_op)) / float(len(nouts_per_op))

    def min_arguments_per_operation(
        self,
        ignore_custom: bool = False,
        vectorized: bool = False,
    ) -> int:
        """
        Compute minimum number of arguments per operation in the model.
        Result will always be at least 1.
        """
        ops = self._operation_nodes(ignore_custom)
        return np.min(
            [nargs(self.flat_graph, op, vectorized) for op in ops])

    def max_arguments_per_operation(
        self,
        ignore_custom: bool = False,
        vectorized: bool = False,
    ) -> int:
        """
        Compute maximum number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the maximum. If Result will always be at
        least 1.
        """
        ops = self._operation_nodes(ignore_custom)
        return np.max(
            [nargs(self.flat_graph, op, vectorized) for op in ops])

    def min_outputs_per_operation(
        self,
        ignore_custom: bool = False,
        vectorized: bool = False,
    ) -> int:
        """
        Compute minimum number of arguments per operation in the model.
        Result will always be at least 1.
        """
        ops = self._operation_nodes(ignore_custom)
        return np.min(
            [nouts(self.flat_graph, op, vectorized) for op in ops])

    def max_outputs_per_operation(
        self,
        ignore_custom: bool = False,
        vectorized: bool = False,
    ) -> int:
        """
        Compute maximum number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the maximum. If Result will always be at
        least 1.
        """
        ops = self._operation_nodes(ignore_custom)
        return np.max(
            [nouts(self.flat_graph, op, vectorized) for op in ops])

    def responses(
        self,
        include: Dict[str, bool] = dict(),
        all: bool = True,
    ) -> List[VariableNode]:
        """
        Gather outputs that correspond to objective and constrants defining an optimization problem. In general, this does not include all outputs.

        include
        : properties that response variables must have to be included

        all
        : whether each response variable must have all properties (True)
        specified in `include` or at least one property (False); default
        is True

        Properties to specify:
        - convex
        - nonconvex
        - linear
        """
        raise NotImplementedError

    def optimization_problem_type(self) -> str:
        """
        Determine optimization problem type. Use output to recommend/choose solver.
        """
        raise NotImplementedError

    def influences(
        self,
        return_promoted_names: bool = True,
        include_objective: bool = True,
        include_constraints: bool = True,
        include_all_outputs: bool = False,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Find all responses influenced by each input, and all inputs that
        influence each response. Returns two dictionaries; one mapping
        each input to all the responses it influences, and one mapping
        each response to all inputs that influence the response. By
        default, promoted names for only the objective and constraints
        are returned.

        **Parameters**

        promoted
        : Each dictionary key value pair contains promoted names of
        variables by default.

        include_objective
        : whether to include the objective in the responses. Defalut is
        True.

        include_constraints
        : whether to include the constraints in the responses. Defalut
        is True.

        include_all_outputs
        : whether to include all outputs, whether or not they are part
        of the optimization problem in the responses. Default is False.
        If True, then `include_objective` and `include_constraints`
        options are ignored

        **Returns**

        Two dictionaries, one mapping inputs to the responses they
        influence, and one mapping responses to the inputs that
        influence them.
        """
        g = self.flat_graph
        registered_outputs: List[
            VariableNode] = get_registered_outputs_from_graph(g)
        map_output_to_inputs: Dict[str, Set[str]] = dict()
        for r in registered_outputs:
            map_output_to_inputs[r.var.abs_prom_name] = set(
                filter(
                    lambda x: x.var.abs_prom_name,
                    filter(lambda x: g.in_degree(x) == 0,
                           ancestors(g, r)),
                ))
        map_input_to_outputs: Dict[str, Set[str]] = dict()
        for r in registered_outputs:
            for leaf in set(
                    filter(lambda x: g.in_degree(x) == 0,
                           ancestors(g, r))):
                if str(leaf) not in map_input_to_outputs.keys():
                    map_input_to_outputs[str(leaf)] = set()
                map_input_to_outputs[str(leaf)].add(r.var.abs_prom_name)
        return map_output_to_inputs, map_input_to_outputs

    def linear_constraints(self) -> Dict[str, List[str]]:
        """
        Return mapping of constraints to the design variables that influence each constraint linearly.
        """
        raise NotImplementedError

    def nonlinear_constraints(self) -> Dict[str, List[str]]:
        """
        Return mapping of constraints to the design variables that
        influence each constraint nonlinearly.
        """
        raise NotImplementedError

    def custom_operations(self) -> List[OperationNode]:
        """
        Return all `CustomExplicitOperation` and
        `CustomImplicitOperation` nodes.
        """
        custom_ops: Iterable[OperationNode] = filter(
            lambda x: isinstance(x.op, (
                CustomExplicitOperation,
                CustomImplicitOperation,
            )),
            self._operation_nodes(False),
        )
        return list(custom_ops)

    def count_operation_types(
        self,
        print: bool = True,
    ) -> Dict[str, int]:
        """
        Count the number of operations of each type and optionally print
        a table to the console (default `True`).

        **Example**

        If `print` is `True`:

        Operation Type      | Number of Nodes |    ??
        ------------------- | --------------- | -----
        sin                 |              30 |
        cos                 |              12 |
        linear_combination  |             189 |
        """
        ops = self._operation_nodes(False)
        optypes: dict[str, int] = dict()
        for op in ops:
            optype = type(op.op).__name__
            if optype not in optypes.keys():
                optypes[optype] = 1
            else:
                optypes[optype] += 1
        title1 = 'Operation Type'
        title2 = 'Number of Nodes'
        if print is True:
            col1_width = max(title1,
                             max([len(x) for x in optypes.keys()]))
            col2_width = max(title2, max(optypes.values()))

        return optypes

    def visualize(self, implicit_models: bool = False):
        """
        Visualize the flattened graph containing variable/operation
        nodes and edges; setting `implicit_models` flag to `True` will
        also visulaize the flattened graph for each model defining an
        implicit operation
        """
        _visualize(self)
        if implicit_models is True:
            apply_fn_to_implicit_operation_nodes(self, _visualize)

    def _operation_nodes(
        self,
        ignore_custom: bool,
    ) -> Iterable[OperationNode]:
        """
        Gathers operation nodes from flattened representation for other
        methods to use.
        """
        operation_nodes: Iterable[OperationNode] = filter(
            lambda x: isinstance(x, OperationNode),
            self.flat_graph.nodes())
        if ignore_custom is False:
            return operation_nodes
        else:
            return filter(
                lambda x: not isinstance(x, (CustomExplicitOperation,
                                             CustomImplicitOperation)),
                operation_nodes)

    def _variable_nodes(self) -> Iterable[VariableNode]:
        """
        Gathers variable nodes from flattened representation for other
        methods to use.
        """
        variable_nodes: Iterable[VariableNode] = filter(
            lambda x: isinstance(x, VariableNode),
            self.flat_graph.nodes())
        return variable_nodes
