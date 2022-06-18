try:
    from csdl.lang.model import Model
except ImportError:
    pass
from typing import Set, List, Tuple


def share_namespace(a: Set[str], b: Set[str]):
    """
    Check if two sets of promoted names create a cycle
    """
    cycles_between_models: List[Tuple[str, str]] = []
    for x in a:
        for y in b:
            if x.rpartition('.')[0] == y.rpartition('.')[0]:
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


def check_for_promotion_induced_cycles(model: 'Model'):
    """
    Check that connections formed by promotions do not create cycles
    between submodels
    """
    for k1, s1 in model.promoted_names_to_unpromoted_names.items():
        for k2, s2 in model.promoted_names_to_unpromoted_names.items():
            if k1 != k2:
                share_namespace(s1, s2)
