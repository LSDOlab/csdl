from typing import Dict, List, Tuple, Set, Union

import numpy as np

from csdl.lang.variable import Variable
from csdl.lang.output import Output
from csdl.utils.get_shape_val import get_shape_val
from csdl.utils.slice_to_list import slice_to_list
from csdl.operations.indexed_passthrough import indexed_passthrough


class Concatenation(Output):
    """
    Class for creating an explicit output
    """

    def __init__(
        self,
        name,
        val=1.0,
        shape: Tuple[int] = (1, ),
        units=None,
        desc='',
        *args,
        **kwargs,
    ):
        """
        Initialize explicit output

        **Parameters**

        name: str
            Name of variable to compute explicitly
        shape: Tuple[int]
            Shape of variable to compute explicitly
        val: Number or ndarray
            Initial value of variable to compute explicitly
        """
        super().__init__(name, *args, **kwargs)
        self.shape, self.val = get_shape_val(shape, val)
        self.units = units
        self.desc = desc

        self.defined: bool = False
        self._tgt_indices: Dict[str, Tuple[Tuple[int],
                                           np.ndarray]] = dict()
        self.checked_indices: Set[np.ndarray] = set()
        self.overlapping_indices: Set[np.ndarray] = set()
        self._tgt_vals: Dict[str, np.ndarray] = dict()

    # TODO: index by tuple, not expression?
    # TODO: allow negative indices
    # TODO: broadcast: expand scalars
    # TODO: broadcast: expand arrays
    def __setitem__(
        self,
        key: Union[int, slice, Tuple[slice]],
        var: Variable,
    ):
        # will generate a list [of lists] of ints, and then convert to
        # ndarray
        tgt_indices: Union[List[int], List[List[int]], np.ndarray] = []

        # n-d array assignment
        if isinstance(key, tuple):
            # allocate list before assignment; initialize with slices
            # to be consistent with type
            newkey: List[slice] = [slice(None, None, None)] * len(key)
            for i, s in enumerate(list(key)):
                if isinstance(s, int):
                    newkey[i] = slice(s, s + 1, None)
                else:
                    newkey[i] = s
            slices = [
                slice_to_list(
                    s.start,
                    s.stop,
                    s.step,
                    size=self.shape[i],
                ) for i, s in enumerate(newkey)
            ]

            tgt_indices = np.ravel_multi_index(
                tuple(np.array(np.meshgrid(*slices, indexing='ij'))),
                self.shape,
            )
        # 1-d array assignment
        elif isinstance(key, slice):
            tgt_indices = slice_to_list(
                key.start,
                key.stop,
                key.step,
                size=self.shape[0],
            )
        # integer index assignment
        elif isinstance(key, int):
            tgt_indices = [key]
        else:
            raise TypeError(
                "When assigning indices of an expression, key must be an int, a slice, or a tuple of slices"
            )

        # Check that source and target shapes match
        if np.array(tgt_indices).shape != var.shape:
            raise ValueError(
                "Shape of LHS {} and RHS {} do not match".format(
                    np.array(tgt_indices).shape, var.shape))
        tgt_indices = np.array(tgt_indices).flatten()

        # Check that indices are in range
        if np.amax(tgt_indices) >= np.prod(self.shape):
            raise ValueError("Indices given are out of range for " +
                             self.__repr__())

        if var.name in self._tgt_indices.keys():
            raise KeyError("Repeated use of expression " + var.name +
                           " in assignment to elements in " +
                           self.name + ". Consider using csdl.expand")
        self._tgt_indices[var.name] = (var.shape, tgt_indices)
        if not hasattr(var, 'val'):
            var.val = np.ones(var.shape)
        self._tgt_vals[var.name] = var.val

        # Check for overlapping indices
        self.overlapping_indices = self.checked_indices.intersection(
            tgt_indices)
        self.checked_indices = self.checked_indices.union(tgt_indices)
        if len(self.overlapping_indices) > 0:
            raise ValueError(
                "Indices used for assignment must not overlap")

            # expr_indices = self._tgt_indices,
            # out_name = self.name,
            # out_shape = self.shape,
            # vals = self._tgt_vals,

        # create indexed passthrough operation to compute this output
        if self.defined is False:
            self.defined = True
            op = indexed_passthrough(
                var,
                output=self,
            )
            self.dependencies = [op]
        else:
            self.dependencies[0].add_dependency_node(var)
