try:
    from csdl.lang.model import Model
except ImportError:
    pass
import enum
from networkx import DiGraph, adjacency_matrix, dag_longest_path, dag_longest_path_length
from typing import List, Dict, Literal, Tuple, Any, Union, List
from csdl.lang.custom_explicit_operation import CustomExplicitOperation
from csdl.lang.custom_implicit_operation import CustomImplicitOperation
from networkx import DiGraph
from csdl.rep.ir_node import IRNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.implicit_operation_node import ImplicitOperationNode
from csdl.rep.model_node import ModelNode
from csdl.rep.construct_unflat_graph import construct_unflat_graph, construct_graphs_all_models
from csdl.rep.construct_flat_graph import construct_flat_graph
from csdl.rep.variable_node import VariableNode
from typing import Dict, List, Set, Tuple
from csdl.rep.ir_node import IRNode
from csdl.rep.sort_nodes_nx import sort_nodes_nx
from csdl.rep.get_registered_outputs_from_graph import get_registered_outputs_from_graph
from csdl.rep.resolve_promotions import resolve_promotions
from csdl.rep.collect_connections import collect_connections
from csdl.rep.collect_design_variables import collect_design_variables
from csdl.rep.collect_constraints import collect_constraints
from csdl.rep.find_objective import find_objective
from csdl.rep.create_graph_with_operations_as_edges import create_graph_with_operations_as_edges
from csdl.lang.bracketed_search_operation import BracketedSearchOperation
from csdl.utils.prepend_namespace import prepend_namespace
from csdl.utils.find_promoted_name import find_promoted_name
from networkx import DiGraph, ancestors, simple_cycles
try:
    pass
except ImportError:
    pass
from csdl.lang.define_models_recursively import define_models_recursively
from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt
from networkx import draw_networkx

from csdl.rep.get_nodes import get_implicit_operation_nodes, get_input_nodes, get_operation_nodes, get_output_nodes, get_var_nodes


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


def _var_sizes(variable_nodes: List[VariableNode], ) -> List[int]:
    """
    Compute the number of scalar values in each vectorized variable in
    `variable_nodes`
    """
    var_sizes = [np.prod(x.var.shape) for x in variable_nodes]
    return var_sizes


def _visualize_unflat(graph: DiGraph):
    """
    Visualize the unflattened graph for the main model containing
    variable/operation nodes and edges; will generate visualization for
    each model in the hierarchy
    """
    model_nodes: List[ModelNode] = list(
        filter(lambda x: x is ModelNode, graph.nodes()))
    for mn in model_nodes:
        _visualize_unflat(mn.graph)
    draw_networkx(graph, labels={n: n.name for n in graph.nodes()})
    plt.show()


def create_reverse_map(d: Dict[str, Set[str]]) -> Dict[str, str]:
    d2: Dict[str, str] = dict()
    for k, v in d.items():
        for kk in v:
            d2[kk] = k
    return d2


def generate_unpromoted_promoted_maps(model: 'Model') -> Dict[str, str]:
    for s in model.subgraphs:
        s.submodel.unpromoted_to_promoted = generate_unpromoted_promoted_maps(
            s.submodel)
    model.unpromoted_to_promoted = create_reverse_map(
        model.promoted_to_unpromoted)
    return model.unpromoted_to_promoted


def structure_user_declared_connections(
    connections: Dict[str, Tuple[dict, List[Tuple[str, str]]]],
    model: 'Model',
) -> Tuple[Dict[str, Tuple[dict, List[Tuple[str, str]]]], List[Tuple[
        str, str]]]:
    for s in model.subgraphs:
        c = dict()
        connections[s.name] = (c, s.submodel.user_declared_connections)

        p, q = structure_user_declared_connections(c, s.submodel)

        assert c is p
        assert s.submodel.user_declared_connections is q

    return connections, model.user_declared_connections


def size(x: IRNode):
    if isinstance(x, VariableNode):
        return np.prod(x.var.shape)
    return 0


def compute_mem(
    A: csc_matrix,
    sorted_nodes: list[IRNode],
    k: int,
    vectorized=True,
):
    if isinstance(sorted_nodes[k], VariableNode):
        return 0
    rows, _ = A.nonzero()
    rows = set(np.unique(rows))
    a: int = 0
    for i, v in enumerate(sorted_nodes[:k]):
        if isinstance(v, VariableNode):
            if i in rows:
                a += np.prod(v.var.shape)

    b: int = sum(
        size(v) * A[k, i + k] for i, v in enumerate(sorted_nodes[k:]))

    return a + b


class GraphRepresentation:
    """
    The intermediate representation of a CSDL Model, stored as a
    directed acyclic graph. The intermediate representation contains
    nodes representing variables and operations. The subgraph node is
    also used to condense graphs representing submodels to encode
    hierarchy. An intermediate representation may also be flattened to
    encode hierarchy without the use of subgraph nodes.
    """
    def __init__(
            self,
            model: 'Model',
            analytics: bool = False,
            rep_name: str = '',
        ):
        self.model_TEMP = model
        check_compilation = False # True to perform common debuggin checks
        self.name = type(model).__name__
        define_models_recursively(model)
        _, _, _, _ = resolve_promotions(model)
        generate_unpromoted_promoted_maps(model)
        connections: List[Tuple[str, str,
                                str]] = collect_connections(model, )
        self.user_declared_connections: Tuple[
            Dict[str, Tuple[dict, List[Tuple[str, str]]]],
            List[Tuple[str,
                       str]]] = structure_user_declared_connections(
                           dict(), model)
        self.connections: List[Tuple[str, str]] = [
            (find_promoted_name(
                prepend_namespace(c, a),
                model.promoted_to_unpromoted,
                model.unpromoted_to_promoted,
            ),
                find_promoted_name(
                prepend_namespace(c, b),
                model.promoted_to_unpromoted,
                model.unpromoted_to_promoted,
            )) for (a, b, c) in connections
        ]
        # remove duplicate connections
        self.connections = list(dict.fromkeys(self.connections))

        self.unpromoted_to_promoted: Dict[
            str, str] = model.unpromoted_to_promoted
        self.promoted_to_unpromoted: Dict[
            str, Set[str]] = model.promoted_to_unpromoted
        
        if check_compilation:
            # check that each value in promoted_to_unpromoted is a key in
            # unpromoted_to_promoted
            for k in self.promoted_to_unpromoted.keys():
                if k not in self.unpromoted_to_promoted.values():
                    raise KeyError(
                        "Promotion maps not found for variable {}. This indicates an error in the compiler implementation, not the user's model."
                        .format(k))

            # Properties of promoted_to_unpromoted/unpromoted_to_promoted
            # -- All values in p2u to are unique
            # -- All keys in p2u are values in u2p
            # -- All keys in u2p are values in p2u
            # -- # of values in p2u == # of keys in u2p

            # check that each unpromoted name in promoted_to_unpromoted is a
            # key in unpromoted_to_promoted
            for k, v in self.promoted_to_unpromoted.items():
                for vv in v:
                    if vv not in self.unpromoted_to_promoted.keys():
                        raise KeyError(
                            "Promotion maps not found for variable {}. This indicates an error in the compiler implementation, not the user's model."
                            .format(k))
                    
            # check that there are no duplicate unpromoted names
            from collections import Counter
            from functools import reduce
            unpromoted_name_sets = list(
                self.promoted_to_unpromoted.values())
            all_unpromoted_names = reduce(
                lambda x, y: x + y, [list(s) for s in unpromoted_name_sets])
            c = Counter(all_unpromoted_names)
            duplicate_unpromoted_names = [
                item for item, count in c.items() if count > 1
            ]
            if len(duplicate_unpromoted_names) > 0:
                raise KeyError(
                    "Found duplicate unpromoted names {}. This indicates an error in the compiler implementation, not the user's model."
                    .format(duplicate_unpromoted_names))

        # collect information about optimization problem
        self.design_variables: Dict[str, Dict[
            str, Any]] = collect_design_variables(
                model,
                model.promoted_to_unpromoted,
                model.unpromoted_to_promoted,
                design_variables=dict()
        )
        """
        Design variables of the optimization problem, if an optimization
        problem is defined
        """
        self.objective: Dict[str, Any] = find_objective(
            model,
            model.promoted_to_unpromoted,
            model.unpromoted_to_promoted,
        )
        """
        Objective of the optimization problem, if an optimization
        problem is defined
        """
        self.constraints: Dict[str,
                               Dict[str, Any]] = collect_constraints(
                                   model,
                                   model.promoted_to_unpromoted,
                                   model.unpromoted_to_promoted,
                                   constraints=dict(),
        )
        """
        Constraints of the optimization problem, if a constrained
        optimization problem is defined
        """
        # TODO: check that there are never multiple sources connected to
        # a single target
        """
        Connections issued to models across model hierarchy branches,
        for use by back ends that use `unflat_graph` only
        """
        # build a graph for each model without child models; this will
        # be used to generate a flat graph and an unflat graph
        first_graph: DiGraph = construct_graphs_all_models(
            model.inputs,
            model.registered_outputs,
            model.declared_variables,
            model.subgraphs,
        )

        all_vars: List[VariableNode] = get_var_nodes(first_graph)
        for v in all_vars:
            name = prepend_namespace(v.unpromoted_namespace, v.name)
            if name in self.promoted_to_unpromoted.keys():
                v.namespace = v.unpromoted_namespace
            elif name in self.unpromoted_to_promoted.keys():
                v.namespace = '.'.join(
                    self.unpromoted_to_promoted[name].rsplit('.')[:-1])

        graph_in = first_graph.copy()
        graph_in.model_nodes = first_graph.model_nodes
        graph_meta = construct_flat_graph(
            graph_in,  # first_graph.copy(),
            connections,
            self.promoted_to_unpromoted,
            self.unpromoted_to_promoted,
            analytics = analytics,
            rep_name = rep_name,
        )
        self.flat_graph: DiGraph = graph_meta.graph
        self.connected_tgt_nodes_to_source_nodes = graph_meta.connected_tgt_nodes_to_source_nodes
        self.promoted_to_node = graph_meta.promoted_to_node
        self.unpromoted_to_node = graph_meta.unpromoted_to_node
        """
        Flattened directed acyclic graph representing main model.
        Only the main model will contain an instance of
        `IntermediateRepresentation` with `flat_graph: DiGraph`.
        All submodels in the hierarchy will contain `flat_graph = None`.
        """
        self.flat_sorted_nodes: List[IRNode] = sort_nodes_nx(
            self.flat_graph,
            flat=True,
        )
        n_nodes = len(self.flat_graph.nodes)
        n_sorted_nodes = len(self.flat_sorted_nodes)
        if n_nodes != n_sorted_nodes:
            raise ValueError(
                "The number of nodes {} is not equal to the number of sorted nodes {}. this indicates a compiler error, not an error in the user's model."
                .format(n_nodes, n_sorted_nodes))
        """
        Nodes sorted in order of execution, using the flattened graph
        """
        # if unflat is True:
        #     self.unflat_graph: DiGraph = construct_unflat_graph(
        #         first_graph)
        #     """
        #     Directed graph representing model.
        #     Each model in the model hierarchy will contain an instance of
        #     `IntermediateRepresentation` with `unflat_graph: DiGraph`.
        #     """
        #     self.unflat_sorted_nodes: List[IRNode] = sort_nodes_nx(
        #         self.unflat_graph,
        #         flat=False,
        #     )
        #     """
        #     Nodes sorted in order of execution, using the unflattened graph
        #     """

        # self._variable_nodes: List[VariableNode] = get_var_nodes(
        #     self.flat_graph)
        # self._operation_nodes: List[
        #     OperationNode] = get_operation_nodes(self.flat_graph)
        # self._std_operation_nodes: List[OperationNode] = [
        #     op for op in self._operation_nodes
        #     if not isinstance(op.op, (CustomExplicitOperation,
        #                               CustomImplicitOperation))
        # ]
        # self.A = adjacency_matrix(self.flat_graph,
        #                           nodelist=self.flat_sorted_nodes)
        # self._density = self.A.nnz / np.prod(self.A.shape)
        # self._longest_path = dag_longest_path(self.flat_graph)
        # self._mem = 0  #self.compute_best_case_memory_footprint()
        # self._critical_path_length = self.compute_critical_path_length()

        implicit_operation_nodes = get_implicit_operation_nodes(
            self.flat_graph)
        # for op in implicit_operation_nodes:
        # op.rep = GraphRepresentation(op.op._model)

        # from csdl.opt.combine_operations import combine_operations
        # combine_operations(self)

    def __str__(self):
        return f"""
GraphRepresentation of model {self.name} Stats:

Scalar Variables (System Level):

Scalar Inputs: {self.num_scalar_inputs()}
Scalar Outputs: {self.num_scalar_outputs()}
Total Scalar Variables: {self.num_scalar_variables()}

Vectorized Variables (System Level):

Input Arrays: {self.num_input_nodes()}
Output Arrays: {self.num_output_nodes()}
Total Variable Arrays: {self.num_variable_nodes()}

Vectorized Operations (System Level):

Total Vectorized Operations: {self.num_operation_nodes()}
Number of Implicit Operations: {self.num_implicit_operation_nodes()}
Average Number of Array Arguments/Operation: {self.avg_arguments_per_operation()}
Maximum Number of Array Arguments/Operation: {self.max_arguments_per_operation()}
Average Number of Array Outputs/Operation: {self.avg_outputs_per_operation()}
Maximum Number of Array Outputs/Operation: {self.max_outputs_per_operation()}

Standard Vectorized Operations (System Level):

Average Number of Array Arguments/Operation: {self.avg_arguments_per_operation(ignore_custom=True)}
Maximum Number of Array Arguments/Operation: {self.max_arguments_per_operation(ignore_custom=True)}
Average Number of Array Outputs/Operation: {self.avg_outputs_per_operation(ignore_custom=True)}
Maximum Number of Array Outputs/Operation: {self.max_outputs_per_operation(ignore_custom=True)}

Adjacency Matrix of Graph Representation (System Level):

Density: {self.density()}
Sparsity: {self.sparsity()}.
Total Nodes: {self.num_variable_nodes() + self.num_operation_nodes()}
Total Edges: {self.A.nnz}

Runtime Time and Space Complexity:

Longest Sequence of Operations to Evaluate (Critical Path): {self.critical_path_length()}
Max Number of Operations to Evaluate: {self.serial_cost()}
Percent Reduction in Computation Time with Full Parallelization: {100*(1 - self.critical_path_length()/self.serial_cost())}
  - NOTE: This assumes all explicit operations take the same amount of time to compute
"""

    def compute_best_case_memory_footprint(
        self,
        vectorized=True,
    ):
        explicit = max(
            compute_mem(self.A,
                        self.flat_sorted_nodes,
                        k,
                        vectorized=vectorized)
            for k in range(len(self.flat_sorted_nodes)))
        x = [
            x.rep.compute_best_case_memory_footprint(
                vectorized=vectorized) for x in [
                    x for x in self.flat_sorted_nodes
                    if isinstance(x, OperationNode)
            ] if isinstance(x, (ImplicitOperationNode))
        ]
        if len(x) == 0:
            self._mem = explicit
        else:
            self._mem = max(explicit, max(x))
        return self._mem

    def longest_path(self) -> List[IRNode]:
        return self._longest_path

    def serial_cost(self) -> int:
        implicit_ops: List[ImplicitOperationNode] = [
            x for x in self.flat_sorted_nodes
            if isinstance(x, ImplicitOperationNode)
        ]
        for op in implicit_ops:
            maxiter = op.op.maxiter if isinstance(
                op.op, BracketedSearchOperation
            ) else op.op.nonlinear_solver.options['maxiter']
            op.cost = op.rep.serial_cost() * maxiter
            print('cost', op.cost)
        explicit_ops = [
            x for x in self.flat_sorted_nodes
            if isinstance(x, OperationNode)
            and not isinstance(x, ImplicitOperationNode)
        ]
        a = len(explicit_ops)
        b = sum([op.cost for op in implicit_ops])
        print(a, b)
        return a + b

    def compute_critical_path_length(self) -> int:
        implicit_ops: List[ImplicitOperationNode] = [
            x for x in self.flat_sorted_nodes
            if isinstance(x, ImplicitOperationNode)
        ]
        if len(implicit_ops) == 0:
            return len([
                x for x in self._longest_path
                if isinstance(x, OperationNode)
            ])
        g = create_graph_with_operations_as_edges(self.flat_graph)
        return dag_longest_path_length(g)

    def critical_path_length(self) -> int:
        return self._critical_path_length

    def input_nodes(self) -> List[VariableNode]:
        """
        Return nodes that represent inputs to the main model.
        """
        return get_input_nodes(self._variable_nodes)

    def output_nodes(self) -> List[VariableNode]:
        """
        Return nodes that represent outputs of the main model.
        """
        return get_output_nodes(self._variable_nodes)

    def operation_nodes(
        self,
        ignore_custom: bool = False,
    ) -> List[OperationNode]:
        """
        Return nodes that represent operations within the model. Uses
        flattened representation to gather operations.
        """
        return self._operation_nodes if ignore_custom is False else self._std_operation_nodes

    def variable_nodes(self) -> List[VariableNode]:
        """
        Return nodes that represent all variables within the model. Uses
        flattened representation to gather variables.
        """
        return self._variable_nodes

    def num_input_nodes(self) -> int:
        """
        Total number of inputs; equivalent to `len(IntermediateRepresentation.input_nodes())`.
        """
        return len(self.input_nodes())

    def num_output_nodes(self) -> int:
        """
        Total number of output arrays
        """
        return len(self.output_nodes())

    def num_scalar_outputs(self) -> int:
        return np.sum(_var_sizes(self.output_nodes()))

    def num_scalar_inputs(self) -> int:
        return np.sum(_var_sizes(self.input_nodes()))

    def num_scalar_variables(self) -> int:
        g: List[ImplicitOperationNode] = [
            o for o in self.flat_sorted_nodes
            if isinstance(o, ImplicitOperationNode)
        ]
        a = sum([o.rep.num_scalar_variables()
                 for o in g]) if len(g) > 0 else 0
        b = np.sum(_var_sizes(self.variable_nodes()))
        return a + b

    def num_implicit_operation_nodes(
        self,
        include: Union[Dict[str, bool], None] = None,
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
        return len([
            x for x in self.operation_nodes()
            if isinstance(x, ImplicitOperationNode)
        ])

    def num_operation_nodes(
        self,
        include: Union[Dict[str, bool], None] = None,
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
    ) -> float:
        """
        Compute average number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the average.
        """
        ops = self._operation_nodes if ignore_custom is False else self._std_operation_nodes
        nargs_per_op = [nargs(self.flat_graph, op, True) for op in ops]
        return float(np.sum(nargs_per_op)) / float(len(nargs_per_op))

    def avg_outputs_per_operation(
        self,
        ignore_custom: bool = False,
    ) -> float:
        """
        Compute average number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the average.
        """
        ops = self._operation_nodes if ignore_custom is False else self._std_operation_nodes
        nouts_per_op = [nouts(self.flat_graph, op, True) for op in ops]
        return float(np.sum(nouts_per_op)) / float(len(nouts_per_op))

    def avg_num_uses_of_variables(
        self,
        ignore_custom: bool = False,
        vectorized: bool = False,
    ) -> float:
        """
        Compute average number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the average.
        """
        ops = self._operation_nodes if ignore_custom is False else self._std_operation_nodes
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
        ops = self._operation_nodes if ignore_custom is False else self._std_operation_nodes
        return np.min(
            [nargs(self.flat_graph, op, vectorized) for op in ops])

    def max_arguments_per_operation(
        self,
        ignore_custom: bool = False,
    ) -> int:
        """
        Compute maximum number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the maximum. If Result will always be at
        least 1.
        """
        ops = self._operation_nodes if ignore_custom is False else self._std_operation_nodes
        return np.max([nargs(self.flat_graph, op, True) for op in ops])

    def min_outputs_per_operation(
        self,
        ignore_custom: bool = False,
        vectorized: bool = False,
    ) -> int:
        """
        Compute minimum number of arguments per operation in the model.
        Result will always be at least 1.
        """
        ops = self._operation_nodes if ignore_custom is False else self._std_operation_nodes
        return np.min(
            [nouts(self.flat_graph, op, vectorized) for op in ops])

    def max_outputs_per_operation(
        self,
        ignore_custom: bool = False,
    ) -> int:
        """
        Compute maximum number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the maximum. If Result will always be at
        least 1.
        """
        ops = self._operation_nodes if ignore_custom is False else self._std_operation_nodes
        return np.max([nouts(self.flat_graph, op, True) for op in ops])

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
        custom_ops: List[OperationNode] = [
            x for x in self._operation_nodes
            if isinstance(x.op, (CustomExplicitOperation,
                                 CustomImplicitOperation))
        ]
        return custom_ops

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
        optypes: dict[str, int] = dict()
        for op in self._operation_nodes:
            optype = type(op.op).__name__
            if optype not in optypes.keys():
                optypes[optype] = 1
            else:
                optypes[optype] += 1
        title1 = 'Operation Type'
        title2 = 'Number of Nodes'
        if print is True:
            col1_width = max(len(title1),
                             max([len(x) for x in optypes.keys()]))
            col2_width = max(len(title2), max(optypes.values()))

        return optypes

    def density(self):
        return self._density

    def sparsity(self):
        return 1 - self._density

    def visualize_adjacency_mtx(
        self,
        markersize=None,
        implicit_models: bool = False,
        title: str = '',
    ):
        """
        Visualize the adjacency matrix for the flattened graph
        representation of the model; setting `implicit_models` flag to
        `True` will also visualize each model defining an implicit
        operation.
        """
        plt.spy(self.A, markersize=markersize)
        ax = plt.gca()
        if title != '':
            plt.title(title)
        t = len(self.flat_sorted_nodes)
        ax.set_xticks([0, t])
        ax.set_yticks([0, t])
        plt.show()

        for i, op in enumerate(self.flat_graph):
            if isinstance(op, ImplicitOperationNode):
                print(i)

        if implicit_models is True:
            for op in get_implicit_operation_nodes(self.flat_graph):
                op.rep.visualize_adjacency_mtx(
                    markersize=markersize,
                    implicit_models=implicit_models,
                )

    def visualize_graph(self):
        draw_networkx(self.flat_graph,
                      labels={
                          n: prepend_namespace(n.namespace, n.name)
                          for n in self.flat_graph.nodes()
                      })
        plt.show()

    # TODO: add implicit models option
    def visualize_unflat_graph(self):
        _visualize_unflat(self.unflat_graph)
