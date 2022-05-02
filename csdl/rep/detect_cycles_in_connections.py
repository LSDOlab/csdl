from networkx import DiGraph, simple_cycles
from typing import List, Tuple


def detect_cycles_in_connections(connections: List[Tuple[str, str]], ):
    """
    Detect cycles formed from user-specified connections. If there are
    cycles present, raise an error.
    """
    g = DiGraph()
    g.add_edges_from(connections)
    cycles_between_models: List[str] = list(simple_cycles(g))
    num_cyles = len(cycles_between_models)
    if num_cyles > 0:
        raise KeyError(
            "Connections resulting from user-specified CONNECTIONS form the following {} cycles between variables: {}. "
            "CSDL does not support cyclic connections between models "
            "to describe coupling between models. "
            "To describe coupling between models, use an implicit operation. "
            "Up to {} implicit operations will be required to eliminate this "
            "error.".format(num_cyles, cycles_between_models,
                            num_cyles))
