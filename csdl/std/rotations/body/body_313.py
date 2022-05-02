from csdl.std import sin as s
from csdl.std import cos as c
from csdl.lang.concatenation import Concatenation
from csdl.lang.variable import Variable
from typing import Tuple


def check_axes(R: Concatenation, axes: Tuple[int]):
    len(axes) == 2


def body_313(
        R: Concatenation,
        p: Variable,
        q: Variable,
        r: Variable,
        axes=(0, 1),
):
    """
    Construct transformation matrix that transforms 3-vector, or array
    of 3-vectors from reference frame A to reference frame B, where
    reference frame B is constructed by rotating reference frame A by a
    body 3-1-3 sequence
    """
    R[0, 0] = c(p) * c(r) - s(p) * c(q) * s(r)
    R[0, 1] = c(p) * s(r) + s(p) * c(q) * c(r)
    R[0, 2] = s(p) * s(q)
    R[1, 0] = -s(p) * c(r) - c(p) * c(q) * s(r)
    R[1, 1] = -s(p) * s(r) + c(p) * c(q) * c(r)
    R[1, 2] = c(p) * s(q)
    R[2, 0] = s(q) * s(r)
    R[2, 1] = -s(q) * c(r)
    R[2, 2] = c(q)
