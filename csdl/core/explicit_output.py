from typing import Dict, List, Tuple, Union, Set

import numpy as np

from csdl.core.variable import Variable
from csdl.core.output import Output
from csdl.utils.get_shape_val import get_shape_val
from csdl.utils.replace_output_leaf_nodes import replace_output_leaf_nodes
from csdl.utils.slice_to_list import slice_to_list
from csdl.core.node import Node
from csdl.operations.passthrough import passthrough
from csdl.operations.indexed_passthrough import indexed_passthrough


class ExplicitOutput(Output):
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
        tags=None,
        shape_by_conn=False,
        copy_shape=None,
        res_units=None,
        lower=None,
        upper=None,
        ref=1.0,
        ref0=0.0,
        res_ref=1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize explicit output

        Parameters
        ----------
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
        self.res_units = res_units
        self.desc = desc
        self.lower = lower
        self.upper = upper
        self.ref = ref
        self.ref0 = ref0
        self.res_ref = res_ref
        self.tags = tags
        self.tags = tags
        self.shape_by_conn = shape_by_conn
        self.copy_shape = copy_shape

        self.defined: bool = False
        self.indexed_assignment: bool = False
        self._tgt_indices: Dict[str, Tuple[Tuple[int], np.ndarray]] = dict()
        self.checked_indices: Set[np.ndarray] = set()
        self.overlapping_indices: Set[np.ndarray] = set()
        self._tgt_vals: Dict[str, np.ndarray] = dict()

    # TODO: automatically create model with NLBGS solver
    def define(self, var: Variable):
        """
        Define expression (in terms of ``self``) that computes value for
        this output. This method defines a cyclic relationship, which
        requires an iterative solver to converge.

        Parameters
        ----------
        var: Variable
            The expression to compute iteratively until convergence
        """
        if var is self:
            raise ValueError(
                "Variable for output {} cannot be self".format(self.name), )
        if self.indexed_assignment == True and self.defined == True:
            raise ValueError(
                "Variable for output {}"
                "is already defined using indexed assignment; use index assignment to concatenate expression outputs"
                .format(self.name))

        if self.defined == True:
            raise ValueError(
                "Variable for output {}"
                ", which forms a cycle to be computed iteratively, is already defined"
                .format(self.name))
        self.defined = True

        # create passthrough operation for backend to recover edges that
        # form cycles
        op = passthrough(var, output=self)
        self.add_dependency_node(op)

        # replace references to this output with references to variable
        # of same name; guarantees that IR is a DAG
        replace_output_leaf_nodes(
            self,
            self,
            Variable(self.name, shape=self.shape, val=self.val),
        )

    # TODO: index by tuple, not expression?
    # TODO: allow negative indices
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
            raise ValueError("Shape of LHS {} and RHS {} do not match".format(
                np.array(tgt_indices).shape, var.shape))
        tgt_indices = np.array(tgt_indices).flatten()

        # Check that indices are in range
        if np.amax(tgt_indices) >= np.prod(self.shape):
            raise ValueError("Indices given are out of range for " +
                             self.__repr__())

        if var.name in self._tgt_indices.keys():
            raise KeyError("Repeated use of expression " + var.name +
                           " in assignment to elements in " + self.name +
                           ". Consider using csdl.expand")
        self._tgt_indices[var.name] = (var.shape, tgt_indices)
        self._tgt_vals[var.name] = var.val

        # Check for overlapping indices
        self.overlapping_indices = self.checked_indices.intersection(
            tgt_indices)
        self.checked_indices = self.checked_indices.union(tgt_indices)
        if len(self.overlapping_indices) > 0:
            raise ValueError("Indices used for assignment must not overlap")

            # expr_indices = self._tgt_indices,
            # out_name = self.name,
            # out_shape = self.shape,
            # vals = self._tgt_vals,

        # create indexed passthrough operation to compute this output
        if self.indexed_assignment is False:
            self.indexed_assignment = True
            op = indexed_passthrough(
                var,
                output=self,
            )
            self.dependencies = [op]
        else:
            self.dependencies[0].add_dependency_node(var)

        self.defined = True
