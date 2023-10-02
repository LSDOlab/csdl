from csdl.lang.variable import Variable
from csdl.utils.get_shape_val import get_shape_val

class DeclaredVariable(Variable):
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
            op=op,
            *args,
            **kwargs,
        )
        self.shape, self.val = get_shape_val(shape, val)

