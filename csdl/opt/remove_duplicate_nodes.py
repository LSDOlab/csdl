from typing import Callable, List, TypeVar, Set, Dict
from csdl.lang.subgraph import Subgraph

from csdl.utils.graph import (
    modified_topological_sort,
    modified_topological_sort_nx,
    # remove_duplicate_nodes,
)
from csdl.rep.graph_representation import GraphRepresentation
from csdl.rep.ir_node import IRNode
from csdl.rep.variable_node import VariableNode
from csdl.rep.operation_node import OperationNode
from csdl.rep.model_node import ModelNode
from csdl.lang.model import Model
from csdl.lang.node import Node
from csdl.lang.concatenation import Concatenation
from csdl.examples.back_end.set_default_values import set_default_values
from networkx import DiGraph

from copy import copy


def remove_duplicate_nodes(graph: DiGraph, sorted_nodes: List[IRNode]):
    """
    Remove nodes resulting from duplicate code in user's model
    definition
    """
    raise NotImplementedError
