from typing import List, Union
import numpy as np


def slice_to_list(
    start: Union[int, None],
    stop: Union[int, None],
    step: Union[int, None],
    size: int = None,
) -> List[int]:
    if start is None and stop is None:
        if size is None:
            raise ValueError("size required when start and stop are None")
        else:
            stop = size
    elif stop is not None:
        if stop < 0:
            stop = size + stop
            print('stop', stop)
            if stop < 0:
                raise ValueError("negative stop index out of range")
    l = list(
        range(
            start if start is not None else 0,
            stop if stop is not None else size,
            step if step is not None else 1,
        ))
    if np.min(l) < 0:
        raise ValueError("negative start index not allowed")
    return l
