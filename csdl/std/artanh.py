from csdl.std.log import log


def artanh(x):
    return 1 / 2 * log((1 + x) / (1 - x))
