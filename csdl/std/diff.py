from csdl.core.concatenation import Concatenation
from csdl.core.variable import Variable


def diff(delta: Concatenation, array: Variable, axis: int):
    """
    Compute the difference between array elements and store in a
    variable of the same size.

    **Example**

    ```py
    T = self.declare_variable('temperature', shape=shape)
    dT = self.create_output('dT', shape=shape, val=0) # val=0 required
    csdl.diff(dT, T, axis)
    ```
    """
    l = array.shape[axis]
    delta_indices = [slice(None, None, None)] * len(array.shape)
    array_indices = [slice(None, None, None)] * len(array.shape)
    sub_indices = [slice(None, None, None)] * len(array.shape)
    # assume value at first index of result is zero
    delta_indices[axis] = slice(1, None, None)
    array_indices[axis] = slice(1, None, None)
    sub_indices[axis] = slice(None, l, None)
    # delta[1:] = array[1:] - array[:l]
    delta[tuple(delta_indices)] = array[tuple(array_indices)] - array[
        tuple(sub_indices)]
