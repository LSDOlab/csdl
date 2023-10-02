from csdl.lang.variable import Variable
from csdl.lang.operation import Operation
from typing import List, Tuple
import numpy as np
from csdl.utils.get_shape_val import get_shape_val

class Output(Variable):
    """
    Base class for outputs; used to prevent circular imports
    """
    def __init__(
        self,
        name,
        val=1.0,
        shape=(1, ),
        units=None,
        desc='',
        op=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            name,
            val=val,
            shape=shape,
            units=units,
            desc=desc,
            *args,
            **kwargs,
        )
        # self.shape, self.val = get_shape_val(shape, val)
        from csdl.lang.concatenation import Concatenation
        if not isinstance(self, Concatenation):
            if not isinstance(op, Operation):
                raise ValueError(
                    "Output object not defined by indexed assignment must depend on an Operation object by construction"
                )
            self.add_dependency_node(op)
