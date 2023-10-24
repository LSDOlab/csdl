from numbers import Number
from typing import Tuple, Union

import numpy as np

from csdl.lang.node import Node
from csdl.lang.subgraph import Subgraph
from csdl.utils.gen_hex_name import gen_hex_name
from csdl.utils.slice_to_list import slice_to_list
from csdl.utils.get_shape_val import get_shape_val

# from csdl.std.binops import (ElementwiseAddition, ElementwiseMultiplication,
# ElementwisePower, ElementwiseSubtraction)

# from csdl.lang.operation import ElementwiseAddition, ElementwiseSubtraction, ElementwiseMultiplication, ElementwiseDivision, ElementwisePower
from csdl.lang.operation import Operation


def slice_to_tuple(key: slice, size: int) -> tuple:
    if key.start is None:
        key = slice(0, key.stop, key.step)
    if key.stop is None:
        key = slice(key.start, size, key.step)
    return (key.start, key.stop, key.step)


class Variable(Node):

    __array_priority__ = 1000
    _unique_id_num = 0

    def __init__(
        self,
        name: str,
        val: np.ndarray = np.array([1.0]),
        shape=(1, ),
        units: Union[str, None] = None,
        desc: str = '',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if name is None:
            self.name: str = self._id
        else:
            self.name: str = name
        self.shape, _ = get_shape_val(shape, val, no_val=True)
        self.size = np.prod(self.shape)
        self.units = units
        self.desc = desc
        self.secondary_name: str = self.name

        self.rep_node = None
        self.rep_nodes = None

        self.default_val = None
        self.unique_id_num = Variable._unique_id_num
        Variable._unique_id_num += 1

        # UNCOMMENT TO DEBUG
        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # rank = comm.rank
        # print(rank, self.unique_id_num, self.name)
        # comm.barrier()

    def add_IR_mapping(self, ir_node:"VariableNode"):
        """
        Used when building the IR.
        Maps this Variable to IR VariableNode(s).
        """

        # Single node
        self.rep_node = {ir_node}

        # Multiple nodes
        if self.rep_nodes is None:
            self.rep_nodes = set()
        self.rep_nodes.add(ir_node)
        # self.rep_node = {ir_node}


    def __pos__(self):
        return self

    def __neg__(self):
        from csdl.operations.linear_combination import linear_combination
        from csdl.lang.output import Output
        op = linear_combination(self, coeffs=-1, constant=0)
        op.outs = [
            Output(
                None,
                op=op,
                shape=op.dependencies[0].shape,
            )
        ]
        return op.outs[0]

    def __add__(self, other):
        from csdl.operations.linear_combination import linear_combination
        from csdl.lang.output import Output
        if isinstance(other, Variable):
            op = linear_combination(self, other, coeffs=1, constant=0)
        elif isinstance(other, (Number, np.ndarray)):
            op = linear_combination(self, constant=other, coeffs=1)
        else:
            raise TypeError(
                "Cannot add {} to an object other than number, NumPy ndarray, or Variable"
                .format(repr(self)))
        op.outs = [
            Output(
                None,
                op=op,
                shape=op.dependencies[0].shape,
            )
        ]
        return op.outs[0]

    def __sub__(self, other):
        from csdl.operations.linear_combination import linear_combination
        from csdl.lang.output import Output
        if isinstance(other, Variable):
            op = linear_combination(self,
                                    other,
                                    constant=0,
                                    coeffs=[1, -1])
        elif isinstance(other, (Number, np.ndarray)):
            op = linear_combination(self, constant=-other, coeffs=1)
        else:
            raise TypeError(
                "Cannot subtract an object from {} other than number, NumPy ndarray, or Variable"
                .format(repr(self)))
        op.outs = [
            Output(
                None,
                op=op,
                shape=op.dependencies[0].shape,
            )
        ]
        return op.outs[0]

    def __mul__(self, other):
        from csdl.operations.power_combination import power_combination
        from csdl.lang.output import Output
        if isinstance(other, Variable):
            op = power_combination(self, other, coeff=1, powers=1)
        elif isinstance(other, (Number, np.ndarray)):
            if isinstance(other, Number):
                op = power_combination(
                    self,
                    out_shape=(1, ),
                    coeff=other,
                    powers=1,
                )

            if isinstance(other, np.ndarray):
                op = power_combination(self, coeff=other, powers=1)
        else:
            raise TypeError(
                "Cannot multiply {} by an object other than number, NumPy ndarray, or Variable"
                .format(repr(self)))
        op.outs = [
            Output(
                None,
                op=op,
                shape=op.dependencies[0].shape,
            )
        ]
        return op.outs[0]

    def __truediv__(self, other):
        from csdl.operations.power_combination import power_combination
        from csdl.lang.output import Output
        if isinstance(other, Variable):
            # TODO: check for near-zero values too
            # if np.any(other.val == 0):
            #     raise ZeroDivisionError(
            #         "Dividing by default zero-valued Variable is guaranteed to cause a divide by zero error at runtime"
            #     )
            op = power_combination(self, other, coeff=1, powers=[1, -1])
        elif isinstance(other, (Number, np.ndarray)):
            # TODO: check for near-zero values too
            # if other == 0 or (isinstance(other, np.ndarray)
            #                   and np.any(other)):
            #     raise ZeroDivisionError(
            #         "Dividing by zero-valued compile time constant is guaranteed to cause a divide by zero error at runtime"
            #     )
            op = power_combination(self, coeff=1 / other, powers=1)
        else:
            raise TypeError(
                "Cannot multiply {} by an object other than number, NumPy ndarray, or Variable"
                .format(repr(self)))
        op.outs = [
            Output(
                None,
                op=op,
                shape=op.dependencies[0].shape,
            )
        ]
        return op.outs[0]

    def __pow__(self, other):
        from csdl.operations.power_combination import power_combination
        from csdl.lang.output import Output
        if isinstance(other, Variable):
            raise NotImplementedError(
                "Raising a variable to a variable power is not yet supported"
            )
        elif isinstance(other, (Number, np.ndarray)):
            op = power_combination(self, coeff=1, powers=other)
        else:
            raise TypeError(
                "Cannot multiply {} by an object other than number, NumPy ndarray, or Variable"
                .format(repr(self)))
        op.outs = [
            Output(
                None,
                op=op,
                shape=op.dependencies[0].shape,
            )
        ]
        return op.outs[0]

    def __radd__(self, other):
        return self.__add__(other)
        # from csdl.operations.linear_combination import linear_combination
        # from csdl.lang.output import Output
        # if isinstance(other, (Number, np.ndarray)):
        #     op = linear_combination(self, constant=other, coeffs=1)
        # else:
        #     raise TypeError(
        #         "Cannot add {} to an object other than number, NumPy ndarray, or Variable"
        #         .format(repr(self)))
        # op.outs = [
        #     Output(
        #         None,
        #         op=op,
        #         shape=op.dependencies[0].shape,
        #     )
        # ]
        # return op.outs[0]

    def __rsub__(self, other):
        from csdl.operations.linear_combination import linear_combination
        from csdl.lang.output import Output
        if isinstance(other, (int, float)):
            op = linear_combination(self, constant=other, coeffs=-1)
        elif isinstance(other, np.ndarray):
            op = linear_combination(self, constant=other, coeffs=-1)

            # raise NotImplementedError(
            #     "Subtraction from NumPy ndarray not yet implemented."
            #     "Instead, change \'a - b\' to \'-(b - a)\'")
        else:
            raise TypeError(
                "Cannot add {} to an object other than number, NumPy ndarray, or Variable"
                .format(repr(self)))
        op.outs = [
            Output(
                None,
                op=op,
                shape=op.dependencies[0].shape,
            )
        ]
        return op.outs[0]

    def __rmul__(self, other):
        return self.__mul__(other)


    def __rtruediv__(self, other):
        from csdl.operations.power_combination import power_combination
        from csdl.lang.output import Output
        if isinstance(other, np.ndarray):
            op = power_combination(self, coeff=other, powers=-1)

            # raise NotImplementedError(
            #     "Dividing NumPy ndarray by Variable object not yet supported."
            #     "Instead, change \'a/b\' to \'b*a**(-1)\'")
        # TODO: check for near-zero values too
        # if np.any(self.val == 0):
        #     raise ZeroDivisionError(
        #         "Dividing by default zero-valued Variable is guaranteed to cause a divide by zero error at runtime"
        #     )
        elif isinstance(other, Number):
            # TODO: check for near-zero values too
            # if other == 0 or (isinstance(other, np.ndarray)
            #                   and np.any(other)):
            #     raise ZeroDivisionError(
            #         "Dividing by zero-valued compile time constant is guaranteed to cause a divide by zero error at runtime"
            #     )
            op = power_combination(self, coeff=other, powers=-1)
        else:
            raise TypeError(
                "Cannot divide {} by an object other than number, NumPy ndarray, or Variable"
                .format(repr(self)))
        op.outs = [
            Output(
                None,
                op=op,
                shape=op.dependencies[0].shape,
            )
        ]
        return op.outs[0]

    def __getitem__(
        self,
        key: Union[int, slice, Tuple[slice]],
    ):
        from csdl.operations.decompose import decompose
        from csdl.lang.output import Output

        if self._getitem_called == False:
            self._getitem_called = True
            self._decomp = decompose(self)

        # store key as a tuple of tuples of ints
        # no duplicate keys are stored
        # NOTE: slices are unhashable, so we can't store slices directly
        if isinstance(key, int):
            if key < 0:
                size = np.prod(self.shape)
                key = ((size + key, size + key + 1, None), )
            else:
                key = ((key, key + 1, None), )
        elif isinstance(key, slice):
            key = (slice_to_tuple(
                key,
                np.prod(self.shape),
            ), )
        elif isinstance(key, tuple):
            l = []
            for i in range(len(key)):
                if isinstance(key[i], slice):
                    l.append(slice_to_tuple(
                        key[i],
                        self.shape[i],
                    ))
                elif isinstance(key[i], int):
                    if key[i] < 0:
                        size = self.shape[i]
                        l.append(
                            slice_to_tuple(
                                slice(size + key[i], size + key[i] + 1,
                                      None),
                                self.shape,
                            ))
                    else:
                        l.append(
                            slice_to_tuple(
                                slice(key[i], key[i] + 1, None),
                                self.shape,
                            ))
            key = tuple(l)
        else:
            raise TypeError(
                "Key must be an int, slice, or tuple object containing slice and/or int objects"
            )

        # return output if reusing key
        if key in self._decomp._key_out_pairs.keys():
            return self._decomp._key_out_pairs[key]

        # Get flat indices from key to define corresponding component
        slices = [
            slice_to_list(s[0], s[1], s[2], size=self.shape[i])
            for i, s in enumerate(list(key))
        ]
        src_indices = np.ravel_multi_index(
            tuple(np.array(np.meshgrid(*slices, indexing='ij'))),
            self.shape,
        ).flatten()

        # Check size
        if np.amax(src_indices) >= np.prod(self.shape):
            raise ValueError("Indices given are out of range for " +
                             self.__repr__())

        # Create and store expression to return
        # TODO: clean up _decomp member names
        if not hasattr(self, 'val'):
            self.val = np.ones(self.shape)
        # self.val = np.ones(self.shape)
        val = self.val[tuple(
            [slice(s[0], s[1], s[2]) for s in list(key)])]
        out = Output(None, op=self._decomp, shape=val.shape, val=val)
        out.val = val
        outs = list(self._decomp.outs)
        outs.append(out)
        self._decomp.outs = tuple(outs)
        self._decomp.nouts = len(self._decomp.outs)
        self._decomp._key_out_pairs[key] = out
        self._decomp.src_indices[out] = src_indices

        return out

    # Required for respecting order in which subsystems are added
    def add_dependency_node(self, dependency):
        if not isinstance(dependency, (Operation, Subgraph)):
            raise TypeError(
                "Dependency of a Variable object must be an Operation object"
            )
        if len(self.dependencies) > 0:
            raise TypeError(
                "A Variable object can only have a single dependency, as only one operation can be an output for a given variable"
            )
        self.dependencies.append(dependency)

        # Add dependency
        # self.dependencies.append(dependency)
        # self._dedup_dependencies()

        # if dependency not in self.dependencies:
        #     self.dependencies.append(dependency)
        # # else:
        # #     raise ValueError(dependency.name, 'is duplicate')

    def flatten(self):
        '''
        Returns a flattened version of itself.
        '''
        from csdl.std.reshape import flatten
        return flatten(self)