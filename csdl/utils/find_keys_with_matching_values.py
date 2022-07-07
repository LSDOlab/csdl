from typing import Callable, Dict, Any, TypeVar, Set

T = TypeVar('T', bound=Any)
U = TypeVar('U', bound=Any)


def find_keys_with_matching_values(
    a: Dict[T, U],
    b: Dict[T, U],
    error: Callable[[T, U, T, U], BaseException] | None = None,
) -> Set[T]:
    """
    Find keys common to two dictionaries with matching values. Keys must
    be of a common type, and values must be of a common second type. If
    error provided, raise error if any values mismatch for any key
    common to both dictionaries.
    """
    c: Dict[T, U] = dict()
    d: Dict[T, U] = dict()
    matching_keys: Set[T] = set()

    # ensure that keys match when iterating over dictionary items
    for k in a.keys():
        if k in b.keys():
            c[k] = a[k]
            d[k] = b[k]

    # iterate over dictionary items with matching keys
    for (k1, v1), (k2, v2) in zip(c.items(), d.items()):
        if v1 == v2:
            matching_keys.add(k1)
        elif error is not None:
            raise error(k1, v1, k2, v2)
        else:
            continue
    return matching_keys
