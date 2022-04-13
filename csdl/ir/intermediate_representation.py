import networkx as nx
from typing import List, Dict, TypeVar, Literal, Tuple
from csdl.core.implicit_operation import ImplicitOperation
from csdl.core.node import Node
from csdl.core.input import Input
from csdl.core.output import Output
from csdl.core.operation import Operation
from csdl.core.variable import Variable
from csdl.core.custom_explicit_operation import CustomExplicitOperation
from csdl.core.custom_implicit_operation import CustomImplicitOperation
import matplotlib.pyplot as plt


class IntermediateRepresentation:
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
        graph: nx.DiGraph,
    ):
        self.unflat_graph: nx.DiGraph = graph
        """
        Directed acyclic graph representing model.
        Each model in the model hierarchy will contain an instance of
        `IntermediateRepresentation` with `unflat_graph: nx.DiGraph`.
        """
        self.unflat_graph_reversed: nx.DiGraph | None = None
        """
        Reversed directed acyclic graph representing
        model; used for modified topological sort function to ignore
        nodes representing dead code.
        Each model in the hierarchy will contain an instance of
        `IntermediateRepresentation` with `unflat_graph: nx.DiGraph`.
        """
        self.flat_graph: nx.DiGraph | None = None
        """
        Flattened directed acyclic graph representing main model.
        Only the main model will contain an instance of
        `IntermediateRepresentation` with `flat_graph: nx.DiGraph`.
        All submodels in the hierarchy will contain `flat_graph = None`.
        """
        self.flat_graph_reversed: nx.DiGraph | None = None
        """
        Reversed flattened directed acyclic graph representing main
        model; used for modified topological sort function to ignore
        nodes representing dead code.
        Only the main model will contain an instance of
        `IntermediateRepresentation` with `flat_graph: nx.DiGraph`.
        All submodels in the hierarchy will contain `flat_graph = None`.
        """

    def flatten_graph(self):
        raise NotImplementedError

    def make_fwd_graph(self, *, flat: bool):
        if flat is True:
            self.flat_graph_reversed = self.flat_graph.reverse()
        else:
            self.unflat_graph_reversed = self.unflat_graph.reverse()

    def optimize(self, options: List[str] | None = None):
        """
        Perform implementation-independent optimizations. This will
        modify `self.graph` and `sorted_nodes`. `options` argument
        specifies which optimizations to perform. If `None`, perform all
        optimizations available

        - Combine elementwise operations to use complex step
          approximation
        - Remove duplicate code

        **Parameters**

        `options` : choose which optimizations to perform. If `None`,
        then perform all available optimizations.
        """
        raise NotImplementedError

    def input_nodes(self) -> List[Input]:
        """
        Return nodes that represent inputs to the main model.
        """
        raise NotImplementedError

    def output_nodes(self) -> List[Output]:
        """
        Return nodes that represent outputs of the main model.
        """
        raise NotImplementedError

    def operation_nodes(self) -> List[Operation]:
        """
        Return nodes that represent operations within the model. Uses
        flattened representation to gather operations.
        """
        raise NotImplementedError

    def variable_nodes(self) -> List[Variable]:
        """
        Return nodes that represent all variables within the model. Uses
        flattened representation to gather variables.
        """
        raise NotImplementedError

    def num_inputs(self) -> int:
        """
        Total number of inputs; equivalent to `len(IntermediateRepresentation.input_nodes())`.
        """
        raise NotImplementedError

    def num_outputs(self) -> int:
        """
        Total number of outputs; equivalent to `len(IntermediateRepresentation.output_nodes())`.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def num_variable_nodes(self) -> int:
        """
        Number of variable nodes; each node represents an n-dimensional
        array specified by user. Equivalent to
        `len(IntermediateRepresentation.variable_nodes())`.
        """
        raise NotImplementedError

    def predict_memory_footprint(
        self,
        mode: Literal['fwd', 'rev'],
    ) -> int:
        """
        Total number of scalar values in model, including variables from
        ImplicitOperation nodes, and residuals required for solving
        implicit operations.
        """
        raise NotImplementedError

    def predict_computation_time(self, parallel: bool = False) -> float:
        """
        Predict computation time to evaluate the model. If parallel is
        false, then computation time is time to evaluate all operations
        in series.  If parallel is True, then computation time for a
        fully parallelized model is equivalent to computation time for
        all operations on the critical path of the graph.
        """
        raise NotImplementedError

    def avg_arguments_per_operation(
        self,
        ignore_custom: bool = False,
    ) -> float:
        """
        Compute average number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the average.
        """
        raise NotImplementedError

    def min_arguments_per_operation(self) -> int:
        """
        Compute minimum number of arguments per operation in the model.
        Result will always be at least 1.
        """
        raise NotImplementedError

    def max_arguments_per_operation(
        self,
        ignore_custom: bool = False,
    ) -> float:
        """
        Compute maximum number of arguments per operation in the model.
        If `ignore_custom` is `False`, then `CustomOperation` nodes are
        not used to compute the maximum. If Result will always be at
        least 1.
        """
        raise NotImplementedError

    def responses(self,
                  include: Dict[str, bool] = None,
                  all: bool = True) -> List[Output]:
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
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
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
        raise NotImplementedError

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

    def custom_operations(
        self
    ) -> List[CustomExplicitOperation | CustomImplicitOperation]:
        """
        Return all `CustomExplicitOperation` and
        `CustomImplicitOperation` nodes.
        """
        raise NotImplementedError

    def count_operation_types(
        self,
        print: bool = True,
    ) -> Dict[str, List[int]]:
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
        raise NotImplementedError

    def visualize(self, mode: str, implicit_models: bool = False):
        """
        Visualize the flattened graph of variable/operation nodes and
        edges; modes argument specifies how to visualize (e.g. `mat`).
        """
        plt.spy(nx.adjacency_matrix(self.flatgraph), markersize=1)
        plt.show()
        if implicit_models is True:
            implicit_operations: List[ImplicitOperation] = list(
                filter(lambda x: isinstance(x, ImplicitOperation),
                       self.sorted_nodes))
            for op in implicit_operations:
                op._model.intermediate_representation.visualize(
                    mode, implicit_models=implicit_models)
