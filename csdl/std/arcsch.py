from csdl.std.log import log


def arcsch(x):
    log(1 / x + (1 / x**2 + 1)**0.5)
