import numpy as np

name_types = (str, list)


def get_names_list(names):
    if isinstance(names, str):
        names = [names]
    elif isinstance(names, list):
        pass
    else:
        raise Exception()

    return names


scalar_types = (int, float, list, np.ndarray)


def get_scalars_list(scalars, names):
    if isinstance(scalars, (int, float)):
        scalars = [scalars] * len(names)
    elif isinstance(scalars, (list)):
        pass
    elif isinstance(scalars, np.ndarray):
        scalars = list(scalars)

    return scalars


shape_types = (tuple, list)


def get_shapes_list(shapes):
    if isinstance(shapes, tuple):
        shapes = [shapes]
    elif isinstance(shapes, list):
        pass
    else:
        raise Exception()

    return shapes
