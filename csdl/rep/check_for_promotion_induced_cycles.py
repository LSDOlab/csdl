try:
    from csdl.lang.model import Model
except ImportError:
    pass
from typing import Set, List, Tuple
from networkx import DiGraph


def share_namespace(a: Set[str], b: Set[str]):
    """
    Check if two sets of promoted names create a cycle
    """
    g = DiGraph()
    g.add_edge(promoted_name_a, promoted_name_b)
    cycles_between_models: List[Tuple[str, str]] = []
    for x in a:
        for y in b:
            # if x and y have the same variable name, but
            # if len(x.rpartition('.')[0]) > 1 or len(
            #         y.rpartition('.')[0]) > 1:
            if x.rpartition('.')[0] == y.rpartition('.')[0]:
                # FIXME: we don't know that these variables are in
                # different models
                cycles_between_models.append((x, y))
    num_cycles = len(cycles_between_models)
    if num_cycles > 0:
        raise KeyError(
            "Connections resulting from user-specified PROMOTIONS form the following {} cycles between variables: {}. "
            "CSDL does not support cyclic connections between models "
            "to describe coupling between models. "
            "To describe coupling between models, use an implicit operation. "
            "Up to {} implicit operations will be required to eliminate this "
            "error.".format(num_cycles, cycles_between_models,
                            num_cycles))
