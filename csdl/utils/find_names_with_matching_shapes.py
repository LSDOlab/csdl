from csdl.utils.find_keys_with_matching_values import find_keys_with_matching_values
from csdl.utils.typehints import Shape
from typing import Callable, Dict, Iterable, List, Tuple, Any, TypeVar, Set


# TODO: warn if sinks have different values, iff they are not connected
def find_names_with_matching_shapes(
    a: Dict[str, Shape],
    b: Dict[str, Shape],
) -> Set[str]:
    """
    Find names of variables whose shapes match. If any shapes mismatch,
    raise an error.
    """
    return find_keys_with_matching_values(
        a,
        b,
        error=lambda name1, name2, shape1, shape2: ValueError(
            "Shapes do not match; {} has shape {}, and {} has shape {}".
            format(name1, name2, shape1, shape2)))
