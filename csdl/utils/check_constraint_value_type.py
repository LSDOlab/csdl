import numpy as np


def check_constraint_value_type(val):
    if val is not None and not isinstance(val,
                                          (int, float, np.ndarray)):
        raise TypeError(
            'Constraint values must be of type int, float, or np.ndarray; {} given'
            .format(type(val)))
