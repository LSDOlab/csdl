from csdl.std.log import log


def arsech(x):
    return log(1 + (1 - x**2)**0.5 / x)
