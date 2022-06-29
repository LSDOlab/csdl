from typing import Callable, Dict, Any, TypeVar, Set

T = TypeVar('T', bound=Any)
U = TypeVar('U', bound=Any)


def check_duplicate_keys(
    a: Dict[str, T],
    b: Dict[str, T],
    # TODO: improve error message
    msg: Callable[[Set[str]], str] = lambda x:
    "Cannot promote two inputs, two outputs, or an input and an output with the same name. Please check the variables with the following promoted paths: {}"
    .format(list(x)),
):
    """
    Check that key names are not duplicated between dictionaries.
    If duplicates are found, raises an error with message `msg`.
    """
    duplicates = b.keys() & a.keys()
    if duplicates != set():
        raise KeyError(msg(duplicates))
