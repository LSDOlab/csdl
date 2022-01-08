from csdl.std.log import log


def arcosh(x):
    return log(x + (x**2 - 1)**0.5)
